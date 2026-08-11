# Steering Language Models With Activation Engineering (ActAdd)

## (a) Full citation + URL

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, Monte MacDiarmid. **"Steering Language Models With Activation Engineering."** arXiv:2308.10248 [cs.CL], 2023 (v3).

- Abstract: https://arxiv.org/abs/2308.10248
- PDF: https://arxiv.org/pdf/2308.10248 (downloaded → `kb/pdf/actadd-2308.10248.pdf`)

## (b) Problem addressed
Introduces **ActAdd**: inference-time activation steering via a single contrast pair (no dataset, no optimization/backprop needed) — add a direction derived from *one* pair of prompts (e.g. `"Love"` vs `"Hate"`) to the residual stream at a chosen layer, at every generation step, to steer sentiment/topic/toxicity without finetuning.

## (c) Method — equations and setup

**Steering vector** (single contrast pair, or averaged over a small set of pairs):
$$h_a^\ell = h_+^\ell - h_-^\ell$$
where $h_+^\ell, h_-^\ell$ are residual-stream activations at layer $\ell$ for the positive/negative contrast prompt respectively (e.g. the token-position-matched activation for the last token of "Love" vs "Hate").

**Injection** (added at every token position, every generation step, at the chosen layer):
$$h^\ell \leftarrow h^\ell + c \cdot h_a^\ell$$
where $c$ is the injection coefficient ("intervention strength"), typically swept up to magnitude ~15. Key finding: **intervening at middle layers is most effective** — directly consonant with the target task's choice of "after the middle layer."

**Models tested**: GPT-2-XL (1.5B) as primary; also demonstrated on LLaMA-3-8B, OPT-6.7B, GPT-J-6B, Llama-1-13B (scaling/generality check).

## (d) Evaluation protocol — exact metrics
- **Toxicity**: Perspective API score on RealToxicityPrompts (n=1000 prompts).
- **Sentiment**: SiEBERT classifier — probability of flipping the sentiment classification, evaluated on IMDb (n=1000).
- **Fluency**: conditional perplexity, computed via GPT-3 `davinci-002` logprobs on the continuation given the (steered-model-generated) prefix.
- **Relevance**: cosine similarity between prompt and completion embeddings (checks the steered model still responds to the actual prompt rather than ignoring it).
- **Knowledge preservation**: P@K on ConceptNet triples (n=29,774 sentences) — checks steering doesn't wreck unrelated factual recall.
- **Topic-specific perplexity ratio**: relative perplexity on wedding-related vs. unrelated OpenWebText sentences (for the "wedding" topic-steering demo).

## (e) Fluency-degradation finding (directly relevant to the task's core problem statement)
The paper explicitly reports: **"fluency is worse under all steering methods; 1.5x–3x worse for ActAdd"** compared to unsteered baselines, and large injection coefficients can produce "unimpressive" or outright incoherent completions. This is essentially the exact failure mode the target task's denoiser is meant to fix — ActAdd is a good "before" baseline to reproduce and then show your `D(h+alpha*v)` recovers fluency at matched (or better) concept strength.

## Directly reusable techniques / pitfalls
1. **Middle-layer injection is empirically best** for GPT-2-XL-scale models — corroborates the task's design choice.
2. **A single contrast pair is sufficient** to define a usable direction (no dataset needed) — useful as a cheap sanity-check steering vector before investing in SAE-feature or persona-vector-style directions.
3. **Report a relevance metric (prompt-completion cosine similarity), not just concept-strength** — guards against the degenerate case where large alpha makes the model ignore the prompt entirely while still scoring "high concept" on a naive classifier.
4. **Always report a knowledge-preservation check (e.g. ConceptNet P@K)** as a second capability-preservation axis beyond perplexity — perplexity alone can look fine while factual recall silently degrades.
5. **Pitfall**: fluency degradation is reported as a fixed multiplicative penalty (1.5–3x) at the coefficients they used for "successful" steering — i.e. some fluency cost is apparently unavoidable with naive additive steering even at moderate (not extreme) alpha; your denoiser's success criterion should be "beats this ActAdd fluency-penalty envelope at matched concept strength," not "zero fluency cost."
