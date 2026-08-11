"""Dump residual-stream activations at HOOK and estimate their first two moments."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from tqdm import tqdm

from common import (
    DATA,
    D_MODEL,
    DEVICE,
    HOOK,
    SKIP_POS,
    ActStats,
    load_model,
    open_memmap,
    seed_all,
)

# OpenWebText is the corpus the gpt2-small-res-jb SAEs were trained on. Shard 0 supplies
# the activation dump; shard 1 supplies generation prompts, so the two never overlap.
CORPUS = "Skylion007/openwebtext"
ACT_SHARD = "hf://datasets/Skylion007/openwebtext/plain_text/train-00000-of-00080.parquet"
PROMPT_SHARD = "hf://datasets/Skylion007/openwebtext/plain_text/train-00001-of-00080.parquet"
TRAIN_DOCS = (0, 12000)
PROMPT_DOCS = (0, 2000)
CTX = 128


def _iter_batches(model, docs, batch_size: int):
    buf = []
    for text in docs:
        if len(text) < 400:
            continue
        toks = model.to_tokens(text[:4000], truncate=True)[0, :CTX]
        if toks.shape[0] < CTX:
            continue
        buf.append(toks)
        if len(buf) == batch_size:
            yield torch.stack(buf)
            buf = []


def dump(n_tokens: int, batch_size: int, seed: int) -> None:
    from datasets import load_dataset

    seed_all(seed)
    model = load_model()
    ds = load_dataset("parquet", data_files=ACT_SHARD, split="train")
    docs = [ds[i]["text"] for i in range(*TRAIN_DOCS)]

    path = DATA / "acts.f16"
    mm = open_memmap(path, n=n_tokens, mode="w+")
    tok_path = DATA / "toks.i32"
    tm = np.memmap(tok_path, dtype=np.int32, mode="w+", shape=(n_tokens,))

    written = 0
    sum_x = torch.zeros(D_MODEL, dtype=torch.float64, device=DEVICE)
    sum_xx = torch.zeros(D_MODEL, D_MODEL, dtype=torch.float64, device=DEVICE)
    norms = []

    pbar = tqdm(total=n_tokens, unit="tok")
    for toks in _iter_batches(model, docs, batch_size):
        if written >= n_tokens:
            break
        with torch.no_grad():
            _, cache = model.run_with_cache(toks.to(DEVICE), names_filter=HOOK)
        h = cache[HOOK][:, SKIP_POS:, :].reshape(-1, D_MODEL)
        tk = toks[:, SKIP_POS:].reshape(-1)
        take = min(h.shape[0], n_tokens - written)
        h, tk = h[:take], tk[:take]
        tm[written : written + take] = tk.cpu().numpy().astype(np.int32)
        h64 = h.double()
        sum_x += h64.sum(0)
        sum_xx += h64.T @ h64
        norms.append(h.norm(dim=-1).float().cpu())
        mm[written : written + take] = h.half().cpu().numpy()
        written += take
        pbar.update(take)
    pbar.close()
    mm.flush()
    tm.flush()
    del mm, tm
    # The memmaps were sized for the requested budget; the corpus may run out first.
    # Truncate so that reading them back never yields zero rows.
    with open(path, "r+b") as fh:
        fh.truncate(written * D_MODEL * 2)
    with open(tok_path, "r+b") as fh:
        fh.truncate(written * 4)

    mean = sum_x / written
    cov = sum_xx / written - torch.outer(mean, mean)
    cov = 0.5 * (cov + cov.T)
    median_norm = float(torch.cat(norms).median())

    stats = ActStats(mean=mean.float(), cov=cov.float(), median_norm=median_norm, n_tokens=written)
    stats.save()

    evals = torch.linalg.eigvalsh(cov).flip(0)
    meta = {
        "corpus": CORPUS,
        "train_docs": TRAIN_DOCS,
        "ctx": CTX,
        "skip_pos": SKIP_POS,
        "n_tokens": written,
        "median_norm": median_norm,
        "trace_cov": float(torch.diagonal(cov).sum()),
        "eig_top10": [float(x) for x in evals[:10]],
        "eig_bottom10": [float(x) for x in evals[-10:]],
        "effective_rank": float(torch.exp(-(lambda p: (p * p.log()).sum())(evals / evals.sum()))),
        "seed": seed,
    }
    (DATA / "act_stats.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def dump_prompts(n_prompts: int, prompt_len: int, seed: int) -> None:
    """Held-out generation prompts, disjoint from the activation corpus."""
    from datasets import load_dataset

    seed_all(seed)
    model = load_model(device="cpu")
    ds = load_dataset("parquet", data_files=PROMPT_SHARD, split="train")
    out = []
    for i in range(*PROMPT_DOCS):
        if len(out) >= n_prompts:
            break
        text = ds[i]["text"].strip()
        if len(text) < 300 or not text[:1].isupper():
            continue
        toks = model.to_tokens(text[:600], truncate=True)[0, 1 : 1 + prompt_len]
        prompt = model.to_string(toks)
        if not prompt.isascii() or "\n" in prompt:
            continue
        out.append(prompt)
    (DATA / "prompts.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{len(out)} prompts, e.g.: {out[:3]}")


def verify_identity() -> None:
    """resid_post[6] must equal resid_pre[7]; the SAE basis argument depends on it."""
    model = load_model()
    toks = model.to_tokens(["The Eiffel Tower stands in the heart of the French capital"])
    with torch.no_grad():
        _, cache = model.run_with_cache(
            toks, names_filter=["blocks.6.hook_resid_post", "blocks.7.hook_resid_pre"]
        )
    a, b = cache["blocks.6.hook_resid_post"], cache["blocks.7.hook_resid_pre"]
    print("max abs diff:", (a - b).abs().max().item())
    assert torch.equal(a, b), "hook identity broken"
    print("identity holds")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["verify", "dump", "prompts"])
    ap.add_argument("--n-tokens", type=int, default=2_000_000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-prompts", type=int, default=40)
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "verify":
        verify_identity()
    elif a.stage == "dump":
        dump(a.n_tokens, a.batch_size, a.seed)
    else:
        dump_prompts(a.n_prompts, a.prompt_len, a.seed)
