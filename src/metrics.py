"""Fluency and concept metrics for generated continuations.

Design choices worth stating:
- Perplexity is measured by an independent scorer (Pythia-410m), not by GPT-2 itself: a broken
  model is confidently wrong about its own output, so self-perplexity does not see the damage.
- It is computed over the continuation tokens only, conditioned on the prompt.
- Prompt dependence, logppl(cont | shuffled prompt) - logppl(cont | true prompt), guards against
  the failure mode where strong steering makes the model ignore the prompt and still score well
  on the concept axis.
- distinct-n follows Li et al. 2016: unique n-grams pooled over a group, divided by that group's
  total token count, at fixed generation length.
- The judge emits a single digit so the score can be read off as a logprob-weighted expectation
  over the ten digit tokens instead of parsing free text.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from common import DEVICE, HOOK, RESULTS, ROOT, load_model, load_sae

SCORER = "EleutherAI/pythia-410m"
SCORER_ALT = "Qwen/Qwen2.5-0.5B"
JUDGE = "Qwen/Qwen2.5-1.5B-Instruct"

JUDGE_TEMPLATE = """You are evaluating a short text continuation produced by a language model.

CONCEPT: {concept}

TEXT:
{text}

On a scale from 0 to 9, how strongly does the TEXT express or reference the CONCEPT above?
0 means the concept is entirely absent. 9 means the text is dominated by the concept.
Answer with a single digit and nothing else."""


def read_gen(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    return pd.DataFrame(rows)


def ngrams(tokens: list[str], n: int):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: list[str], n: int) -> float:
    total, uniq = 0, set()
    for t in texts:
        tok = t.split()
        total += len(tok)
        uniq.update(ngrams(tok, n))
    return len(uniq) / max(total, 1)


def repetition_rate(text: str, n: int = 4) -> float:
    g = ngrams(text.split(), n)
    if not g:
        return 0.0
    return 1.0 - len(set(g)) / len(g)


@torch.no_grad()
def score_ppl(df: pd.DataFrame, model_name: str, batch: int, shuffled: bool = False) -> np.ndarray:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(DEVICE)
    lm.eval()

    prompts = list(df["prompt"])
    if shuffled:
        rng = random.Random(0)
        pool = list(dict.fromkeys(prompts))
        mapping = {p: rng.choice([q for q in pool if q != p] or pool) for p in pool}
        prompts = [mapping[p] for p in prompts]
    texts = list(df["text"])

    out = np.zeros(len(df), dtype=np.float64)
    for i in tqdm(range(0, len(df), batch), unit="batch", leave=False):
        ps, ts = prompts[i : i + batch], texts[i : i + batch]
        enc_p = [tok(p, add_special_tokens=False)["input_ids"] for p in ps]
        enc_t = [tok(t, add_special_tokens=False)["input_ids"] for t in ts]
        seqs = [p + t for p, t in zip(enc_p, enc_t)]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.bool)
        for j, (s, p) in enumerate(zip(seqs, enc_p)):
            ids[j, : len(s)] = torch.tensor(s)
            mask[j, len(p) : len(s)] = True
        ids, mask = ids.to(DEVICE), mask.to(DEVICE)
        logits = lm(ids).logits[:, :-1].float()
        tgt = ids[:, 1:]
        # logsumexp minus the target logit, so the full log-softmax is never materialised
        nll = torch.logsumexp(logits, dim=-1) - logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        m = mask[:, 1:]
        out[i : i + len(seqs)] = (
            ((nll * m).sum(-1) / m.sum(-1).clamp_min(1)).cpu().numpy().astype(np.float64)
        )
        del logits, nll
    del lm
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def score_sae_activation(df: pd.DataFrame, batch: int) -> np.ndarray:
    """Mean activation of each row's target latent when its own text is re-encoded."""
    model = load_model()
    sae = load_sae()
    out = np.zeros(len(df), dtype=np.float64)
    order = df.sort_values("feature").index.to_numpy()
    pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
    positions = {k: p for p, k in enumerate(df.index)}
    for i in tqdm(range(0, len(order), batch), unit="batch", leave=False):
        idx = order[i : i + batch]
        sub = df.loc[idx]
        feats = torch.as_tensor(sub["feature"].to_numpy(), device=DEVICE)
        toks = model.to_tokens(list(sub["text"]))
        _, cache = model.run_with_cache(toks, names_filter=HOOK)
        z = sae.encode(cache[HOOK])
        pick = z[torch.arange(len(idx), device=DEVICE), :, feats]
        # to_tokens prepends BOS and right-pads; both must stay out of the average
        valid = toks != pad_id
        valid[:, 0] = False
        pick = (pick * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        out[[positions[k] for k in idx]] = pick.cpu().numpy()
        del cache, z
    del model, sae
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def score_judge(df: pd.DataFrame, concepts: dict[int, str], batch: int) -> np.ndarray:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(JUDGE)
    lm = AutoModelForCausalLM.from_pretrained(JUDGE, torch_dtype=torch.float16).to(DEVICE)
    lm.eval()
    digit_ids = torch.tensor([tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)])
    digit_ids = digit_ids.to(DEVICE)
    values = torch.arange(10, device=DEVICE, dtype=torch.float32)

    prompts = [
        tok.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": JUDGE_TEMPLATE.format(
                        concept=concepts[int(r.feature)], text=r.text.strip()
                    ),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in df.itertuples()
    ]
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    out = np.zeros(len(df), dtype=np.float64)
    for i in tqdm(range(0, len(prompts), batch), unit="batch", leave=False):
        enc = tok(prompts[i : i + batch], return_tensors="pt", padding=True).to(DEVICE)
        logits = lm(**enc).logits[:, -1, :].float()
        sub = torch.log_softmax(logits, dim=-1)[:, digit_ids]
        w = torch.softmax(sub, dim=-1)
        out[i : i + enc["input_ids"].shape[0]] = (w * values).sum(-1).cpu().numpy()
        del logits, sub, w
    del lm
    torch.cuda.empty_cache()
    return out


def score_keywords(df: pd.DataFrame, keywords: dict[int, list[str]]) -> np.ndarray:
    hits = np.zeros(len(df), dtype=np.float64)
    for i, r in enumerate(df.itertuples()):
        kws = keywords[int(r.feature)]
        prompt_low = r.prompt.lower()
        text_low = r.text.lower()
        fresh = [k for k in kws if k not in prompt_low]
        hits[i] = float(any(k in text_low for k in fresh))
    return hits


def load_feature_meta(split: str) -> tuple[dict, dict]:
    import yaml

    recs = yaml.safe_load((ROOT / "configs" / "features.yaml").read_text(encoding="utf-8"))[split]
    kw = {int(r["index"]): [k.lower() for k in r["keywords"]] for r in recs}
    concept = {int(r["index"]): r.get("concept") or ", ".join(r["keywords"][:6]) for r in recs}
    return kw, concept


def main(args) -> None:
    df = read_gen(RESULTS / args.gen)
    kw, concept = load_feature_meta(args.split)
    stages = args.stages.split(",")

    if "ppl" in stages:
        df["logppl"] = score_ppl(df, SCORER, args.batch)
        df["logppl_shuf"] = score_ppl(df, SCORER, args.batch, shuffled=True)
        df["prompt_dependence"] = df["logppl_shuf"] - df["logppl"]
    if "ppl_alt" in stages:
        df["logppl_alt"] = score_ppl(df, SCORER_ALT, args.batch)
    if "keyword" in stages:
        df["keyword_hit"] = score_keywords(df, kw)
    if "sae" in stages:
        df["sae_act"] = score_sae_activation(df, args.batch)
    if "judge" in stages:
        df["judge"] = score_judge(df, concept, args.judge_batch)
    df["rep4"] = [repetition_rate(t) for t in df["text"]]

    out = RESULTS / args.out
    df.to_csv(out, index=False, encoding="utf-8")

    if "dist" in stages:
        rows = []
        for (arm, feat, c), g in df.groupby(["arm", "feature", "c"]):
            texts = list(g["text"])
            rows.append(
                {
                    "arm": arm,
                    "feature": feat,
                    "c": c,
                    **{f"dist{n}": distinct_n(texts, n) for n in (1, 2, 3)},
                }
            )
        pd.DataFrame(rows).to_csv(RESULTS / args.out.replace(".csv", "_dist.csv"), index=False)
    print(f"scored {len(df)} rows -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="gen_test.jsonl")
    ap.add_argument("--out", default="scored_test.csv")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--stages", default="ppl,keyword,sae,dist")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--judge-batch", dest="judge_batch", type=int, default=16)
    main(ap.parse_args())
