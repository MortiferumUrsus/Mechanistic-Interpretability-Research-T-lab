# Persona Vectors: Monitoring and Controlling Character Traits in Language Models

## (a) Full citation + URL

Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, Jack Lindsey. **"Persona Vectors: Monitoring and Controlling Character Traits in Language Models."** arXiv:2507.21509 [cs.CL, cs.LG]. Submitted 29 Jul 2025, latest v3 5 Sep 2025. Affiliations: Anthropic Fellows Program, UT Austin, Constellation, Truthful AI, UC Berkeley, Anthropic.

- Abstract: https://arxiv.org/abs/2507.21509
- PDF: https://arxiv.org/pdf/2507.21509 (downloaded → `kb/pdf/persona-vectors-2507.21509.pdf`, text-extracted and verified)
- Code: https://github.com/safety-research/persona_vectors (Apache 2.0)

## (b) Problem addressed

LLM "assistant" personas can drift unexpectedly — via in-context prompting/many-shot jailbreak-style prompts, or via **finetuning-induced personality shifts** (including "emergent misalignment," where finetuning on a narrow flawed-but-not-obviously-malicious dataset, e.g. buggy code, causes broad undesirable trait shifts like evil/sycophantic/hallucinating behavior). The paper's goal: (1) an **automated pipeline** (no manual labels beyond a trait name + one-sentence description) to extract a single linear direction — a "persona vector" — per trait; (2) use that direction to *monitor* trait expression via projection, to *steer* the model to induce/suppress the trait, and to *predict/flag* which finetuning data will cause a bad shift, before finetuning.

## (c) Method — exact equations, architecture, hyperparameters

### Extraction pipeline (fully automated, Section 2)
1. **Artifact generation**: given only a trait name + short description, a frontier LLM (Claude 3.7 Sonnet, thinking mode, budget=5000, max_tokens=16000 — from repo README) is prompted once with a fixed meta-template (`data_generation/prompts.py`, key `PROMPTS["generate_trait"]` — full text pulled below) to produce, per trait: (i) **5 pairs of contrastive system prompts** (`pos`/`neg`, each pair independently phrased to elicit/suppress the trait), (ii) **40 evaluation questions** (kept generic — they do not mention the trait, they're meant to *elicit* trait-relevant behavior indirectly), split 20/20 into an **extraction set** and an **eval set**, and (iii) one **trait-specific judge/eval prompt** (0–100 scale, using a fixed template — see judge-prompts note file).
2. **Response generation**: for each of the 5 pos/neg system-prompt pairs × 20 extraction-set questions, sample 10 rollouts each with the target model under the pos system prompt and under the neg system prompt.
3. **Filtering**: keep only responses whose trait-expression judge score (by GPT-4.1-mini) is ≥50 for the pos condition and <50 for the neg condition (both must also pass a coherence filter — see `get_persona_effective` in `activation_steer`/`generate_vec.py`, threshold=50 on both trait score and coherence score).
4. **Vector computation**: extract residual-stream activations at **every layer**, average over response tokens (not prompt tokens — see pitfall #2 below). Persona vector at layer $\ell$:
   $$v_\ell = \text{mean}_{\text{filtered pos}}\big[\bar h_\ell^{\text{resp}}\big] - \text{mean}_{\text{filtered neg}}\big[\bar h_\ell^{\text{resp}}\big]$$
   i.e. a **difference-of-means** vector (mean response-token activation under trait-eliciting system prompt minus mean under trait-suppressing system prompt), computed independently per layer. This yields one candidate vector per layer; the best layer is chosen empirically by testing steering effectiveness across layers (their Appendix B.4 layer sweep).
   Three activation-pooling variants are actually saved by `generate_vec.py`: `prompt_avg_diff.pt`, `response_avg_diff.pt` (**the one used in the paper**), `prompt_last_diff.pt` — each of shape `[n_layers, hidden_dim]`.

### Steering equation (Section 3.2)
$$h_\ell \leftarrow h_\ell + \alpha \cdot v_\ell$$
applied at **every decoding step**, at a single chosen layer $\ell$, where $\alpha$ is a scalar steering coefficient and $v_\ell$ is the (raw, un-normalized) persona vector at that layer. This is the exact same additive-steering equation as the target task. Experiments sweep $\alpha \in$ roughly $\{0.5, 1.0, 1.5, 2.0, 2.5\}$ (Fig. 3's "S steering coefficient" legend) across layers 5–25 (Qwen2.5-7B-Instruct has 28 layers; sweep covers early-to-late-middle layers). In the released repo's example commands, coefficients of 2.0–5.0 are used (`--coef 2.0` for inference-time steering, `steering_coef: 5.0` for training-time preventative steering).

### Implementation (`activation_steer.py`, verbatim logic)
`ActivationSteerer` is a forward-hook context manager: it locates the model's transformer block list (tries `transformer.h`, `model.layers`, etc., for GPT-2/Llama-style models respectively), registers a forward hook on `layer_idx` that adds `coeff * steering_vector` to the block's *output* hidden state. `positions` controls which token positions get the addition: `"all"` (every position, every step), `"prompt"` (prompt tokens only), or `"response"` (**only the last position each step**, i.e. the currently-generated token) — the paper's main results use `steering_type="response"`. Note: `layer_idx` passed to the hook is `layer-1` relative to the "layer" index used elsewhere (see `sample_steering(..., layer-1, ...)` in `eval/eval_persona.py`) — an off-by-one convention to be careful about when porting.

### Models used
Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct (main text); results transfer across both. Three focal traits in main text: **evil**, **sycophantic**, **hallucinating** (+4 more incl. optimistic/humorous in appendix — see `data_generation/trait_data_eval/` for all 7 trait json files: apathetic, evil, hallucinating, humorous, impolite, optimistic, sycophantic).

### Training-time "preventative steering" (Section 5)
To stop finetuning from inducing unwanted persona shifts without hurting deployment behavior: add the steering vector **during training only** (not at inference), configured via JSON (`configs/train_instruct_7b_steer.json`):
```json
{
  "enable_steering_during_training": true,
  "steering_config": {
    "steering_vector_path": "persona_vectors/model/trait_response_avg_diff.pt",
    "type": "steer",
    "steering_coef": 5.0,
    "layers": [20]
  }
}
```
`type: "steer"` adds the vector during the forward pass at train time (a preventative regularizer — the idea being if the model is *already* being pushed toward the trait direction internally, the marginal gradient signal from bad training data has less new "room" to push it further); `type: "ablate"` instead implements CAFT-style projection-ablation (remove the component along the trait direction). LoRA config used in main experiments: rank 32, alpha 64, LR 1e-5, batch size 2/device, grad accumulation 8 steps.

### Projection / monitoring metric (`eval/cal_projection.py`)
For monitoring (not steering), they compute a scalar **projection** of an activation onto the (unit-normalized) persona direction:
$$\text{proj}(a, v) = \frac{a \cdot v}{\lVert v \rVert_2}$$
(function `a_proj_b` in the repo — note this is a *projection*, i.e. scaled by $\lVert v\rVert$ only, not a cosine similarity; `cos_sim` is also provided separately as `(a·v)/(\lVert a\rVert \lVert v\rVert)`). Computed either on response-token-average activation or on the last prompt-token activation (`projection_type` = `"proj"` / `"prompt_last_proj"` / `"cos_sim"`).

## (d) Evaluation protocol — exact metrics

Two LLM-judge scores computed per generated response, **both required together** for every eval:
1. **Trait expression score** (0–100): trait-specific judge prompt (see `kb/notes/persona-vectors-judge-prompts.md` for the verbatim template) — generated per-trait by the extraction pipeline itself, always instructs the judge to output `REFUSAL` or a 0–100 integer.
2. **Coherence score** (0–100): a single shared, trait-independent judge prompt (`eval/prompts.py`, `Prompts["coherence_0_100"]`) — full text in the judge-prompts note file. 0 = "not even valid English," 50 = "rambling," 100 = "clearly understood the question, no hallucinations/confusion." Explicitly **does not judge correctness/ethics**, only coherence — this is their fluency-analog metric, directly relevant as your fluency axis.

Judge model: **GPT-4.1-mini** (`gpt-4.1-mini-2025-04-14`) by default, called via OpenAI API. Scoring mechanism (`judge.py`, class `OpenAiJudge`) is **not** free-text parsing — it requests `max_tokens=1, logprobs=True, top_logprobs=20, temperature=0, seed=0` and computes a **probability-weighted expectation over the top-20 logprob tokens** that parse as integers in range:
$$\text{score} = \frac{\sum_{k: \text{int}(k)\in[0,100]} k \cdot p(k)}{\sum_{k: \text{int}(k)\in[0,100]} p(k)}$$
returning `None` (treated as refusal/invalid, excluded from means) if the combined probability mass on valid numeric tokens is `< 0.25`. This logprob-weighted-average trick avoids the variance of single-sample discrete scoring and is a **directly reusable pattern** for any 0–100 LLM-judge metric.

Standard eval command (`eval/eval_persona.py`, batched over `n_per_question=100` samples per question, both baseline (`coef=0`) and steered (`coef≠0`, requires `--vector_path` + `--layer`)):
```bash
python -m eval.eval_persona --model <hf_model_or_path> --trait evil \
    --output_path out.csv --judge_model gpt-4.1-mini-2025-04-14 --version eval \
    [--coef 2.0 --vector_path persona_vectors/<model>/evil_response_avg_diff.pt --layer 20 --steering_type response]
```
Output CSV has one row per (question, sample) with columns `question, prompt, answer, question_id, <trait>, coherence`; script prints `mean ± std` per trait and for coherence.

## (e) Directly reusable techniques / pitfalls

1. **Always report a paired (concept, coherence) score, never concept alone** — their coherence judge prompt is a ready-made, battle-tested fluency-proxy prompt you can reuse verbatim as an LLM-judge fluency metric alongside perplexity/dist-n, especially useful because perplexity alone can be fooled by degenerate-but-low-perplexity repetition, while an LLM coherence judge catches rambling/nonsense more directly.
2. **Extract the concept/steering vector from *response*-token activations, not prompt-token activations** — they found this gives more effective steering directions (footnote 2, Section 2.2), a concrete, validated design choice for wherever you compute your training/eval activations for D.
3. **Filter your contrastive extraction set by both trait-score AND coherence-score thresholds (both ≥50) before computing the mean-difference vector** — don't include low-coherence "trait-eliciting" samples in the vector computation, they contaminate the direction with generic incoherence rather than pure trait signal.
4. **Use the logprob-weighted 0–100 scoring trick** (single-token sample, `top_logprobs=20`, weighted average over parseable integers, `None` if <0.25 mass) instead of parsing a free-text numeric answer — cheaper (1 token), lower-variance, and has a principled REFUSAL/invalid-response handling path built in.
5. **Steer only the last (currently-generated) token position per decoding step** (`positions="response"`) rather than all positions — this was their primary configuration and avoids compounding the additive shift across the whole KV-cache history.
6. **Preventative/training-time steering as an alternative to post-hoc/inference-time steering** — if your denoiser D is meant to fix fluency, also consider: does *always* denoising (even absent an explicit alpha*v perturbation) act as a regularizer during any fine-tuning you might do, analogous to their `steering_config.type="steer"` during training?
7. **Pitfall**: their `ActivationSteerer` hook adds the vector to the block's *raw output* (post-block, i.e. `hook_resid_post`-equivalent) — if you're steering at "after the middle layer" per the task, make sure your hook point matches whichever hook_resid_post/pre convention your denoiser was trained on (see the SAELens and TransformerLens notes for the resid_pre vs resid_post distinction and why they're numerically identical at adjacent layer boundaries).
8. **Pitfall**: layer/coefficient choice is not free — the paper explicitly runs a layer-sweep to find "the most informative layer" per trait (Appendix B.4) rather than assuming middle-layer is universally optimal; budget for a small layer sweep even if the task fixes "the middle layer" as the injection point, e.g. verify layer 6 is in fact where GPT-2 small responds best to steering before over-optimizing D there.
