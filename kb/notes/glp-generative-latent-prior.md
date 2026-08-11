# Learning a Generative Meta-Model of LLM Activations (Generative Latent Prior, GLP)

## (a) Full citation + URL

Grace Luo, Jiahai Feng, Trevor Darrell, Alec Radford, Jacob Steinhardt. **"Learning a Generative Meta-Model of LLM Activations."** arXiv:2602.06964 [cs.LG, cs.AI, cs.CL], submitted 6 Feb 2026. Accepted ICML 2026.

- Abstract: https://arxiv.org/abs/2602.06964
- PDF: https://arxiv.org/pdf/2602.06964 (downloaded → `kb/pdf/glp-2602.06964.pdf`, verified, text-extracted)
- HTML: https://arxiv.org/html/2602.06964v1
- Project page: https://generative-latent-prior.github.io/
- Official code: https://github.com/g-luo/generative_latent_prior (Apache-licensed PyTorch implementation, ICML 2026)
- HuggingFace weights: https://huggingface.co/generative-latent-prior
- Affiliations: UC Berkeley (Luo, Feng†, Darrell), Independent (Radford), Transluce (Steinhardt). † work done while at UC Berkeley.

**THIS IS THE KEY BASELINE for the task.** It is essentially the exact same problem framing: train a generative denoiser over residual-stream activations and use it to "fix" activations that have been pushed off-manifold by an additive steering intervention, evaluated as a concept-vs-fluency Pareto tradeoff.

## (b) Problem addressed

Activation steering (adding a concept direction to the residual stream) is a simple, effective control method, but larger steering coefficients push activations **off the data manifold**, degrading fluency/coherence. Existing analysis tools (PCA, SAEs) impose strong structural assumptions (linearity, sparsity) that don't capture the true activation distribution and don't fix off-manifold artifacts. GLP's thesis: learn the actual generative distribution of activations with a diffusion model, then use that learned prior at inference time to project a steered (off-manifold) activation back onto the manifold while preserving the intended semantic shift — directly the same "denoiser to recover fluency while keeping concept strength" idea as the target task, just with a diffusion/flow-matching model instead of a bespoke denoising network trained via `||h - D(h+eps)||^2`.

## (c) Method — exact equations, architecture, hyperparameters

### Flow-matching training objective
Forward process (linear interpolation between clean activation and Gaussian noise):
$$z_t = (1-t) z_0 + t\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I),\ t \in [0,1] \qquad \text{(Eq. 1)}$$

Reverse process, given a learned velocity estimate $\hat u$, discretely integrated from $t$ down to $t' < t$:
$$z_{t'} = z_t + \hat u \cdot (t' - t) \qquad \text{(Eq. 2)}$$

The denoiser network $\hat u_\theta(z_t, t)$ is trained to regress the **target velocity** $u = \epsilon - z_0$ (i.e., noise minus clean data — standard rectified-flow / flow-matching parameterization). Loss is plain MSE between predicted and target velocity:
$$\mathcal{L}(\theta) = \mathbb{E}_{z_0,\epsilon,t}\big[\lVert \hat u_\theta(z_t,t) - (\epsilon - z_0) \rVert_2^2\big]$$
Pseudocode confirms: `pred_velocity = denoiser(noisy_acts, timesteps=t)`, `target_velocity = noise - acts`, `loss = MSE(pred_velocity, target_velocity)`.

Activations are standardized to zero mean / unit variance (per-dimension scaler fit on training data) before training/sampling and de-normalized after.

### Architecture
- Denoiser = stack of feedforward **SwiGLU MLP blocks with residual connections**, following the Llama-3 block design (no attention — GLP models **single-token** activations independently, removing the need for attention, similarly to how SAEs treat tokens independently).
- **Unconditional** model (no class/task labels).
- **Timestep conditioning**: multiplicative modulation of the SwiGLU gate pre-activation at each MLP block (the only diffusion-specific architectural change vs. a plain MLP).
- Width = 2× the activation dimension; the SwiGLU's internal gated-MLP expansion factor is an *additional* 2× over that width (so ~4× overall vs. `d_model`). Paper notes (citing Li et al. 2024) that making the GLP "sufficiently wide relative to the input activations" was critical for generation quality.
- Concrete sizes trained (Table 1):
  - **Llama1B** activations (`d=2048`): 0.5B params/3 layers, 0.9B/6 layers, 1.7B/12 layers, 3.3B/24 layers.
  - **Llama8B** activations (`d=4096`): 3.4B params/6 layers (single size used for downstream steering experiments).

### Training data / pipeline
- Source corpus: **FineWeb** (same web corpus commonly used for LLM pretraining).
- **1 billion residual-stream activation tokens**, sampled from all token positions in each document (excluding the BOS token), documents truncated to max length 2048 tokens.
- Activations taken from the **middlemost layer** of the source LLM: Layer 7 of Llama1B, Layer 15 of Llama8B — i.e., exactly analogous to "after the middle layer" in the target task (GPT-2 small, 12 layers → layer 6).
- Custom producer/consumer data pipeline (fixed-size buffer) to avoid the runtime/memory tradeoff of caching-on-the-fly vs. caching-everything; activation caching sped up via vLLM and `nnsight`.
- **Training hyperparameters**: 1 epoch over the 1B activations, batch size 4096, learning rate 5e-5, cosine LR schedule, warmup ratio 0.01, mixed precision. Trained on a single A100-80GB GPU; the largest run (Llama8B, 3.4B params) took 5.6 days.

### Inference-time "on-manifold steering" algorithm (the core reusable trick — Figure 4 pseudocode, transcribed verbatim)
Given a raw steered activation `acts_edit = acts + alpha * w` (w = steering direction, e.g. a persona vector / SAE decoder column / DiffMean vector):

```python
# apply intervention to activations
acts_edit = acts + alpha * w

# standardize to zero mean & unit variance
acts_edit = (acts_edit - scaler.mean) / scaler.std

# noise activations according to pre-specified t_start
# bigger t_start = stronger correction from diffusion sampling
noise = np.random.normal()
acts_noisy = (1 - t_start) * acts_edit + t_start * noise

# init sampling at t=t_start from acts_noisy, instead of t=1 from pure noise
acts_sample = acts_noisy

# run multi-step sampling
timesteps = np.linspace(t_start, 0, num_steps)
for i in range(len(timesteps) - 1):
    t  = timesteps[i]
    dt = timesteps[i+1] - timesteps[i]
    pred_velocity = denoiser(acts=acts_sample, timesteps=t)
    acts_sample = acts_sample + dt * pred_velocity

# restore back to original mean & variance
acts_sample = (acts_sample * scaler.std) + scaler.mean
```

This is exactly an activation-space analog of **SDEdit** (Meng et al. 2022, image editing): instead of starting the reverse diffusion from pure noise (t=1), you initialize from the *off-manifold* steered activation at an **intermediate** noise level `t_start`. `t_start` controls a tradeoff: larger t_start → more noise added → GLP has more freedom to "correct" the activation (stronger fluency recovery, but risks erasing the steering signal); smaller t_start → less correction, stays closer to the raw steered activation.

- **Default hyperparameters used across the steering experiments: `t_start = 0.5`, `num_steps = 20`.** (Same values also used for the "Delta LM Loss" reconstruction-style evaluation.)
- For the Fréchet-Distance generation-quality check (unconditional generation from pure noise, not steering), 1000 diffusion steps were used; 20 steps was empirically sufficient for PCA/Fréchet-Distance convergence when starting from noise.

### How the steering coefficient α is set
"We observe that the steering vector often needs a norm similar to or greater than that of the activation." They therefore parameterize with a **relative coefficient r** and compute the absolute steering coefficient as
$$\alpha = r \cdot \lVert \bar a \rVert_2$$
where $\lVert \bar a \rVert_2$ is the **average activation norm computed from a validation set**. This is directly reusable: instead of hand-picking a raw alpha, normalize by the empirical residual-stream norm at your steering layer and sweep r.

## (d) Evaluation protocol — exact metrics

1. **Representation Fréchet Distance (FD)** between 50k generated (from pure noise) vs. 50k real activations (Dowson & Landau 1982) — measures whether GLP's *unconditional generative* distribution matches the true activation distribution. Lower bound computed as FD(train-set vs. val-set) to quantify irreducible sampling error. Table 1 results: Llama1B lower bound 0.22, SAE-reconstruction 1.99, GLP (24 layers, 3.3B) 0.53 (best); Llama8B lower bound 2.60, SAE-reconstruction 6.91, GLP (6 layers, 3.4B) 5.93 (best).
2. **Delta LM Loss** (standard SAE reconstruction metric, Bricken et al. 2023 / Lieberum et al. 2024): increase in the source LLM's perplexity/loss when original activations at the target layer are replaced by *reconstructed* ones (for GLP: activations noised to `t_start=0.5` then denoised with `num_steps=20`, i.e. reconstruction-via-denoising rather than autoencoding). Evaluated on 2048 held-out OpenWebText sequences (max length 128), all tokens except BOS. GLP beats a comparable SAE: Llama8B-Base 0.0513 (GLP) vs 0.1976 (SAE); Llama8B-Instruct (transfer, no retraining) 0.0860 (GLP) vs 0.2224 (SAE).
3. **Concept-vs-Fluency Pareto front** (the exact evaluation shape requested by the target task): sweep the steering coefficient r/alpha, and at each point plot (fluency score, concept score) as judged by an **LLM-as-judge**. Two scales used depending on experiment:
   - **0–2 scale** (LlamaScope SAE-feature steering experiments, Fig. 5, and DiffMean sentiment steering, Fig. 2b) — methodology follows Wu et al. 2025.
   - **0–100 scale** (persona-elicitation experiments, Fig. 6) — methodology follows Chen et al. 2025 (== the persona_vectors paper/repo you are also collecting; same judge-prompt style: 0 = trait absent, 100 = trait strongly present, judged by an LLM).
   - For SAE-feature steering, "concept score" is graded against the feature's own **Neuronpedia** description (Lin, 2023) — i.e., LLM-judge is asked whether the generated text matches the human-readable Neuronpedia description of the steered SAE feature.
4. **1-D linear probing AUC** on 113 binary concept-classification tasks (Kantamneni et al. 2025): does a single scalar feature (raw neuron / SAE latent / GLP "meta-neuron") linearly separate a binary concept? Protocol: (i) use the heuristic of Gurnee et al. 2023 on the train split to shortlist candidate neurons/features; (ii) fit 1-D logistic-regression probes (L-BFGS, 1000 iters, L2 reg. swept over $\{10^{-5},...,10^0\}$ via 5-fold CV) on each candidate; (iii) pick the best candidate by validation AUC; (iv) report test AUC. Results (Table 4): Llama1B — SAE 0.70, raw layer 0.77, raw MLP neuron 0.79, **GLP 0.84**; Llama8B — SAE 0.76, raw layer 0.77, raw MLP neuron 0.82, **GLP 0.87**. "Meta-neurons" = the internal SwiGLU-gate activations of the GLP denoiser itself at a hyperparameter-chosen timestep t (for the reported "cleanest scaling trend" results, t=0.5), extracted from a **single forward pass** (not full sampling).
5. **Scaling-law fit**: diffusion loss $L(C) = E + A\cdot C^{-\alpha}$ vs. compute $C = 6ND$ (Kaplan et al. 2020 convention); fitted $E=0.52$ (irreducible error floor), $A=435.1$, $\alpha=0.169$ — i.e., loss gap to floor halves roughly every 60× increase in compute. Both the steering Pareto-front quality and the 1-D probing AUC empirically track this same power law, supporting "diffusion loss as a reliable proxy for downstream utility" — i.e., you can use training loss alone to decide when your denoiser is "good enough" without re-running the full steering eval every time.

### Steering experiment matrix (Section 4)
| Experiment | Concept vector source | Steering model | Fluency/concept scoring |
|---|---|---|---|
| SAE feature steering (Fig. 5) | 500 random directions from LlamaScope SAE decoder (He et al. 2024), trained on Llama8B-Base | Llama8B-Base | LLM judge 0–2 scale; concept graded vs. Neuronpedia feature description |
| Persona elicitation (Fig. 6, Table 3) | Persona Vectors (Chen et al. 2025) for traits Evil / Sycophantic / Hallucinating | Llama8B-**Instruct** (GLP itself trained only on Llama8B-**Base** acts — demonstrates base→instruct transfer) | LLM judge 0–100 scale (persona_vectors-style judge) |
| Sentiment steering scaling study (Fig. 2b) | DiffMean vectors (Marks & Tegmark 2024 / Belrose 2023 / Wu et al. 2025) on the controllable-sentiment-generation task of Liu et al. 2021 | Llama1B (multiple GLP sizes) | LLM judge 0–2 scale; also a SetFit 5-point sentiment classifier and conditional NLL, in supplementary figures |

Key qualitative result they emphasize repeatedly: "GLP post-processing expands the Pareto frontier outward" — at matched fluency, +GLP gets higher concept score than the raw steering vector alone (and than SAE-steering alone). Table 3 shows a concrete matched-fluency comparison: Persona Vector alone needs α=2.0 to hit fluency=34, while raw-Persona-Vector+GLP needs the *larger* α=3.0 to hit the *same* fluency=34 — but produces visibly more on-topic/on-trait completions, i.e. GLP lets you push α further before fluency collapses.

## (e) Directly reusable techniques / pitfalls

1. **SDEdit-style partial denoising is the central trick to steal**: don't denoise from pure noise; initialize the reverse process directly from the *steered* activation at an intermediate noise level `t_start`, then run only `num_steps` (~20) of denoising. This is a much cheaper and more direct analog of your `D(h + alpha*v)` than training a full generative model — but the mechanism (partial forward-noise, partial reverse-denoise, one shot) is exactly the shape of the "interpolation t*h+(1-t)*eps" variant your task proposes. `t_start` is your knob for "how much do we trust the denoiser vs. the raw input" — sweep it like a hyperparameter alongside alpha, it is not a fixed constant.
2. **Parameterize alpha as a multiple of the average activation norm** ($\alpha = r\cdot\lVert\bar a\rVert_2$), not as a raw absolute number — this makes alpha sweeps comparable across layers/models and is exactly what you need to build the Pareto front x-axis in a principled way.
3. **Standardize activations (zero-mean/unit-variance) before feeding to the denoiser**, de-normalize after. Skipping this is a likely silent-failure mode: diffusion/denoising objectives assume roughly-Gaussian, unit-scale inputs, and raw GPT-2 residual streams have highly non-uniform per-dimension scale (they'd need this even more than Llama since GPT-2 uses LayerNorm not RMSNorm, so scale is not automatically ~unit).
4. **Only model single-token activations independently (no cross-token attention needed)** — matches how SAEs are trained and is exactly the setup implied by `h~ = D(h + alpha*v)`; don't over-engineer the denoiser architecture with sequence modeling, a widened residual MLP suffices.
5. **Report Delta-LM-Loss AND a Pareto front, not perplexity alone** — Delta LM Loss isolates "does swapping in the reconstructed/denoised activation break the model" from "does the concept intervention work," which maps directly onto your two axes (fluency / concept).
6. **Pitfall they surface**: the denoiser must be "sufficiently wide relative to the input activation dimension" for generation quality (empirically ≥2× width, ≥4× effective MLP expansion) — an under-parameterized denoiser will underfit the activation manifold and won't actually help fluency. For GPT-2 small's `d_model=768`, this suggests width ≥1536 and MLP hidden ≥3072 as a reasonable floor, scaled down from their 2×/4× ratios.
7. **Pitfall / limitation they flag explicitly**: GLP is *unconditional* — it doesn't condition on the un-noised/clean activation, only on the noisy one plus timestep. They note conditioning on the clean activation could reduce information loss for interventions like steering — i.e., a conditional denoiser $D(h+\alpha v \mid h)$ (which is close to what your task's `||h - D(h+\epsilon)||^2` training setup already gives you, since your D sees the un-noised target during training) may be a strict improvement over their unconditional formulation — worth exploiting since you're not bound to their exact recipe.
8. They also note diffusion/GLP loss on a token can be used to **flag unusual/out-of-distribution activations** (an anomaly-detection use), and that multi-token / sequence-level modeling is future work — both relevant if you want to extend beyond single-token steering later.

## Verification notes
All details above (equations, pseudocode, hyperparameters, table numbers) were transcribed directly from the arXiv PDF (pages 1–10, sections 1–6), not from a secondary summarizer — high confidence. Appendix (steering-config Table 9, multi-layer variant in Appendix B.1, additional traits in Appendix G) was not fully read; if exact per-experiment hyperparameter tables are needed later, re-fetch `kb/pdf/glp-2602.06964.pdf` pages beyond 10. `[UNVERIFIED]`: exact page count / total length of paper beyond what was read.
