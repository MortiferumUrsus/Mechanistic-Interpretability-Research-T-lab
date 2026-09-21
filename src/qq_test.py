"""ActAdd's per-token Q-Q diagnostic (Turner et al., arXiv:2308.10248, section 3.1).

The published check for "perplexity improved for the wrong reason": take a held-out corpus,
score every real next token's log-probability under teacher forcing, once clean and once under
an intervention, group the (intervention - clean) delta by token identity, and look at the
tails. A concept-relevant intervention should have an upper tail of tokens that are plausibly
about the concept; if the tail is instead a handful of unrelated high-frequency tokens, the
perplexity gain is an artifact, not evidence the intervention did what it claims.

Run against three directions per feature, at the same steering strength `s`:
  decoder  the raw SAE decoder direction v_hat -- what "naive" steering already uses
  shared   v_hat blended with the corpus-wide correction direction d_bar (kappa = 0.75)
  d_bar    d_bar injected on its own, with no feature direction at all

`d_bar` is not expected to be concept-relevant for any single feature (it is shared across all
of them by construction), so its Q-Q tail is the sharpest test of whether this diagnostic would
actually catch a spurious perplexity win.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats as sp_stats
from tqdm import tqdm

from common import (
    DATA,
    DEVICE,
    HOOK,
    RESULTS,
    ROOT,
    SHARED_DIRECTION,
    load_ceilings,
    load_features,
    load_model,
    load_sae,
    natural_strength,
)
from steering import HookState, make_hook, naive as steer_naive

BATCH_SIZE = 8
MIN_COUNT = 20
TOP_K = 15
QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def _escape(text: str) -> str:
    """Newlines break the one-row-per-token CSV and the NOTE bullet list alike."""
    return text.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")


def collect_docs(model, n_docs: int, ctx: int, batch_size: int = BATCH_SIZE) -> list[torch.Tensor]:
    """Held-out documents from PROMPT_SHARD, starting past the 2000 already used as prompts.

    Same tokenization as activations.py's prompt/activation dumps: `to_tokens(text[:4000],
    truncate=True)`, keep only documents whose token count reaches `ctx`, then hard-truncate.
    """
    from datasets import load_dataset

    from activations import PROMPT_DOCS, PROMPT_SHARD

    ds = load_dataset("parquet", data_files=PROMPT_SHARD, split="train")
    start = PROMPT_DOCS[1]
    toks_list: list[torch.Tensor] = []
    i = start
    with tqdm(total=n_docs, unit="doc", desc="collecting docs") as pbar:
        while len(toks_list) < n_docs and i < len(ds):
            text = ds[i]["text"]
            i += 1
            toks = model.to_tokens(text[:4000], truncate=True)[0]
            if toks.shape[0] < ctx:
                continue
            toks_list.append(toks[:ctx])
            pbar.update(1)
    if len(toks_list) < n_docs:
        raise RuntimeError(
            f"only found {len(toks_list)} documents with >= {ctx} tokens past doc {start} "
            f"(requested {n_docs}); shard exhausted at doc {i}"
        )
    return [torch.stack(toks_list[j : j + batch_size]) for j in range(0, len(toks_list), batch_size)]


@torch.no_grad()
def score_batches(
    model,
    batches: list[torch.Tensor],
    vocab_size: int,
    direction: torch.Tensor | None = None,
    s: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced log-prob of every real next token, accumulated per token id.

    Never materialises logits for the whole corpus: each batch's [B, T, V] logits are reduced to
    a [vocab] sum and a [vocab] count immediately, so peak memory is one batch, not the corpus.
    """
    sum_lp = torch.zeros(vocab_size, dtype=torch.float64, device=DEVICE)
    count = torch.zeros(vocab_size, dtype=torch.int64, device=DEVICE)
    state = HookState()
    hooks = [(HOOK, make_hook(steer_naive, direction, s, state))] if direction is not None else []
    for toks in batches:
        toks = toks.to(DEVICE)
        state.reset()
        if hooks:
            with model.hooks(fwd_hooks=hooks):
                logits = model(toks)
        else:
            logits = model(toks)
        logprobs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = toks[:, 1:]
        gathered = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        flat_targets = targets.reshape(-1)
        flat_lp = gathered.reshape(-1).double()
        sum_lp.index_add_(0, flat_targets, flat_lp)
        count.index_add_(0, flat_targets, torch.ones_like(flat_targets, dtype=torch.int64))
        del logits, logprobs, gathered
    return sum_lp.cpu(), count.cpu()


def token_table(model, sum_base, count_base, sum_interv) -> pd.DataFrame:
    """Per-token base_mean / delta_mean, restricted to tokens seen more than MIN_COUNT times."""
    mask = count_base > MIN_COUNT
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return pd.DataFrame(columns=["token_id", "token_str", "count", "base_mean", "delta_mean"])
    base_mean = (sum_base[idx] / count_base[idx]).numpy()
    delta_mean = (sum_interv[idx] / count_base[idx]).numpy() - base_mean
    token_ids = idx.numpy()
    strs = model.to_string(idx.unsqueeze(-1))  # rank-2 -> one decoded string per row, not one joined string
    strs = [_escape(s) for s in strs]
    df = pd.DataFrame(
        {
            "token_id": token_ids,
            "token_str": strs,
            "count": count_base[idx].numpy(),
            "base_mean": base_mean,
            "delta_mean": delta_mean,
        }
    )
    return df.sort_values("delta_mean").reset_index(drop=True)


def quantile_row(delta_mean: np.ndarray) -> dict:
    row = {}
    if delta_mean.size == 0:
        for q in QUANTILES:
            p = round(q * 100)
            row[f"q{p}"] = float("nan")
            row[f"qnorm{p}"] = float("nan")
        return row
    q_delta = np.quantile(delta_mean, QUANTILES)
    mu = float(delta_mean.mean())
    sigma = float(delta_mean.std(ddof=1)) if delta_mean.size > 1 else 0.0
    q_normal = sp_stats.norm.ppf(QUANTILES, loc=mu, scale=sigma) if sigma > 0 else np.full(len(QUANTILES), mu)
    for q, dv, nv in zip(QUANTILES, q_delta, q_normal):
        p = round(q * 100)
        row[f"q{p}"] = float(dv)
        row[f"qnorm{p}"] = float(nv)
    return row


def format_tail(df: pd.DataFrame, ascending: bool, n: int = TOP_K) -> str:
    if df.empty:
        return ""
    sub = df.sort_values("delta_mean", ascending=ascending).head(n)
    return " | ".join(f"{r.token_str}:{r.delta_mean:+.3f}" for r in sub.itertuples())


def main(args) -> None:
    t0 = time.time()
    model = load_model()
    sae = load_sae()
    ceilings = load_ceilings()
    d_bar = torch.load(SHARED_DIRECTION, map_location=DEVICE)["d_bar"].to(DEVICE).float()
    feats = load_features(args.split)
    feats = feats[: args.features]
    vocab_size = model.cfg.d_vocab

    batches = collect_docs(model, args.n_docs, args.ctx)
    n_docs_got = sum(b.shape[0] for b in batches)
    print(f"collected {n_docs_got} docs of ctx={args.ctx} in {len(batches)} batches")

    sum_base, count_base = score_batches(model, batches, vocab_size)
    total_count = int(count_base.sum().item())
    print(f"base pass: {total_count} scored positions")

    summary_rows = []
    note_parts = [f"# Q-Q diagnostic (ActAdd sec. 3.1), split={args.split}, c={args.c}, "
                  f"n_docs={n_docs_got}, ctx={args.ctx}\n"]

    for rec in feats:
        f = int(rec["index"])
        v = sae.W_dec[f].detach().float()
        v_hat = v / v.norm()
        w = v_hat + 0.75 * d_bar
        w = w / w.norm()
        directions = {"decoder": v_hat, "shared": w, "d_bar": d_bar}
        s = args.c * natural_strength(sae, ceilings, f)

        note_parts.append(f"\n## feature {f}\n")
        for dname, direction in directions.items():
            sum_interv, _ = score_batches(model, batches, vocab_size, direction=direction, s=s)
            mean_delta_all = float((sum_interv.sum() - sum_base.sum()).item() / total_count)

            tdf = token_table(model, sum_base, count_base, sum_interv)
            tdf.to_csv(RESULTS / f"{args.out_prefix}_tokens_{f}_{dname}.csv", index=False)

            top_hi = format_tail(tdf, ascending=False)
            top_lo = format_tail(tdf, ascending=True)

            row = {
                "feature": f,
                "direction": dname,
                "mean_delta_all": mean_delta_all,
                "n_tokens_kept": len(tdf),
                "top15_high": top_hi,
                "top15_low": top_lo,
            }
            row.update(quantile_row(tdf["delta_mean"].to_numpy()))
            summary_rows.append(row)

            note_parts.append(f"\n### {dname}\n")
            note_parts.append(f"mean_delta_all = {mean_delta_all:+.4f} nats, n_tokens_kept = {len(tdf)}\n")
            note_parts.append(f"- top 15 (upper tail): {top_hi}\n")
            note_parts.append(f"- bottom 15 (lower tail): {top_lo}\n")

            print(f"feature {f:6d}  {dname:8s}  mean_delta_all={mean_delta_all:+.4f}  n_tokens_kept={len(tdf)}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS / f"{args.out_prefix}_summary.csv", index=False)
    (RESULTS / f"{args.out_prefix}_NOTE.md").write_text("".join(note_parts), encoding="utf-8")

    print(f"wrote {args.out_prefix}_summary.csv, {args.out_prefix}_NOTE.md, "
          f"and {len(summary_rows)} per-token CSVs to {RESULTS}")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", dest="n_docs", type=int, default=500)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--features", type=int, default=3)
    ap.add_argument("--split", default="test_r3", choices=["test", "dev", "test_r3", "test_r4"])
    ap.add_argument("--out-prefix", dest="out_prefix", default="qq")
    main(ap.parse_args())
