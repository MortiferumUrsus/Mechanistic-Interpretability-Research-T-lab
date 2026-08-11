# Steering Llama 2 via Contrastive Activation Addition (CAA)

## (a) Full citation + URL

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner. **"Steering Llama 2 via Contrastive Activation Addition."** arXiv:2312.06681 [cs.CL], Dec 2023 (latest v4, Jul 2024). Presented at ACL 2024.

- Abstract: https://arxiv.org/abs/2312.06681
- PDF: https://arxiv.org/pdf/2312.06681 (downloaded → `kb/pdf/caa-rimsky-2312.06681.pdf`)

## (b) Problem addressed
Refines ActAdd-style steering by using **many** contrastive multiple-choice pairs (not one) to compute a lower-variance mean-difference steering vector per behavior (e.g. sycophancy, corrigibility, hallucination, myopia, survival-instinct, refusal), evaluated systematically across layers and coefficients on Llama 2 Chat.

## (c) Method — exact equations

**Steering vector** (mean-difference, "MD" vector), averaged over a dataset $D$ of contrastive multiple-choice question pairs, where each pair differs only in which answer letter (A/B) is chosen to match vs. oppose the target behavior:
$$v_{MD} = \frac{1}{|D|} \sum_{(p,c_p,c_n) \in D} \big[a_L(p, c_p) - a_L(p, c_n)\big]$$
where $a_L(p, c)$ is the residual-stream activation at layer $L$ for prompt $p$ with completion-choice-letter $c$ (activation taken at the position of the answer-letter token). This is a **difference-of-means over many pairs**, the direct generalization of ActAdd's single-pair difference — and structurally identical to persona_vectors' response-token mean-difference vector (see `kb/notes/persona-vectors.md`).

**Application during generation**: the vector is added, scaled by a scalar multiplier, **to every token position of the generated text after the end of the initial prompt** (i.e. not to the prompt tokens, only to newly generated tokens — same "response-only" positional convention as persona_vectors' `positions="response"`... actually CAA applies to *all* generated positions, closer to persona_vectors' `positions="all"` restricted to the response span).

**Models & layers**: Llama 2 7B-Chat and 13B-Chat. Optimal layers found via a systematic per-layer sweep: **layer 13** for 7B, **layers 14–15** for 13B — i.e., again solidly "middle layer" for both model sizes, consistent with ActAdd's and the target task's middle-layer choice. Multipliers tested at ±1 and ±2 (small, interpretable range — contrast with SAE-feature-steering papers that sweep much larger relative coefficients).

## (d) Evaluation protocol — exact metrics
1. **Multiple-choice behavioral probability**: average probability the (steered) model assigns to the answer letter matching the target behavior, over 50 held-out test questions per behavior dataset.
2. **Open-ended generation, LLM-judged**: GPT-4 rates open-ended answers 1–10 on how strongly they display the target behavior.
3. **General-capability preservation**: MMLU, reformatted as A/B letter-choice, reporting average probability assigned to the correct answer — used to check steering doesn't collaterally damage general capability. Reported finding: "with some variation, intervention does not significantly affect MMLU" at their tested multipliers (±1/±2) — but note they did **not** systematically push multipliers into the fluency-breaking regime the target task cares about, so this paper's fluency-preservation result should not be over-read as "steering is free" — it's evidence only at modest coefficients.

## Directly reusable techniques / pitfalls
1. **Average over many contrastive pairs, not one** — substantially reduces direction noise vs. ActAdd's single-pair approach; cheap to do (just more forward passes) and should be a default even for a toy GPT-2-small setup.
2. **Do a per-layer coefficient sweep before committing to "the middle layer"** — both CAA and ActAdd empirically re-derive that middle layers work best rather than assuming it; worth a quick confirmatory sweep on GPT-2 small (12 layers) even though layer 6 is a very plausible default.
3. **Always pair a behavioral/concept metric with a general-capability metric (MMLU-style)** — a capability metric is a complementary, task-orthogonal fluency/capability check distinct from perplexity or an LLM coherence judge; consider adding a small held-out LM-capability proxy (e.g. a simple cloze/completion accuracy set) alongside perplexity and dist-n.
4. **Pitfall**: CAA's own coefficient range (±1/±2) is much smaller than what SAE-feature-steering or persona-vector papers use (often r≥1 relative to activation norm, i.e. coefficients that can be 10s in absolute terms) — don't assume CAA's "no MMLU damage" result transfers to the large-alpha, fluency-breaking regime that motivates the denoiser task; that regime is exactly where ActAdd and the SAE-feature-steering literature (see `kb/notes/norm-preserving-steering-fluency.md`) report real degradation.
