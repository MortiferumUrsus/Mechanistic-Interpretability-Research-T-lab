# Track 1 (Mechanistic Interpretability): solution plan

Version 2. Incorporates external reviews of the plan v1 ([reviews/](reviews/)); v1 is preserved in the git history.
Task: [TASK.md](TASK.md)

---

## 0. Requirements and criteria

**Mandatory:**

| # | Task requirement | Where |
|---|---|---|
| R1 | A small LM + vectors `v` for validation | §2 |
| R2 | Fluency and concept metrics; dist-1/2/3 explicitly | §4 |
| R3 | Validation of `h+αv`, reproduction of the Pareto plot | §7, E1 |
| R4 | The denoiser does not know the validation `v` | §3 |
| R5 | Comparison with ordinary steering only after training | §7 |
| R6 | Intervention after the middle layer | §2.1 |
| R7 | Report + repository + checkpoint in a public repository | §8 |
| R8 | Ideas more interesting than the ones described, with evidence | §5 |
| R9 | Analysis: why and how the method works | §6 |

**Grading criteria (reconstructed).** High weight: novelty of the idea with evidence (a direct quote from
the authors) and analysis of the mechanism ("the second half of the paper"). Medium weight: correctness of the
comparisons and control of leakage (visible from R4/R5), honesty of the metrics (visible from the emphasis on the two
axes), code cleanliness and reproducibility. Low weight, but binary: a checkpoint in a public repository.

**The main trap.** The construction `h̃ = denoiser(h+αv)` is given by the authors as an *example* ("the above
is only an example"). Implementing it literally is not enough: see §5.1 — it has two defects, one of which
makes the point `α=0` depend on an arbitrary `v`. Work that finds, explains and fixes this
covers both high-weight criteria.

---

## 1. Preregistration: hypotheses and predictions

Recorded before any runs on TEST. The numbers are predictions, not results.

**H1 (conditional, not unconditional).** In the limit the MSE denoiser gives `D(x) = E[h | h+ε = x]`, that is
`D(x) − x = σ²∇log p_σ(x)`. Hence the sign of the correction along `v` is determined by the local gradient
of the density, and not by "off-manifoldness" as such. In the linear-Gaussian approximation
`D(x) = μ + A(x−μ)`, `A = Σ(Σ+σ²I)⁻¹`, and the **steering transmission** equals

```
τ(v̂, σ) = 1 − σ² · v̂ᵀ(Σ + σ²I)⁻¹ v̂
```

For an eigendirection of `Σ` with variance `s`: `τ = s/(s+σ²)`. Prediction: erasure
`1−τ` is large where the variance of the activations along `v̂` is small against `σ²`, and small where it is large.
Numerically I expect `τ ∈ [0.3, 0.8]` for SAE directions at the σ chosen on DEV.

**H2 (why the naive denoiser does not help).** Under erasure `1−τ` the naive denoiser gives up part of the
concept in exchange for fluency, that is, it yields a point close to `M1` at strength `τ·α`. Prediction: the
`M3` front lies within noise of the `M1` front, not above it.

**H3 (method).** Contrastive direction-preserving correction (`M4`, §5.2) achieves repair without paying in
concept → Pareto dominance over `M1` and `M3`. Prediction: a gain on the primary endpoint of
+10…+30% relative to `M1`.

**H4 (mechanism, closed form).** The main part of the repair is shrinkage of energy in the low-variance
directions of `Σ`. Corollary: `M5` (Mahalanobis transport, requires no training, §5.3) should take
a noticeable share of `M4`'s gain. Prediction: `M5` takes ≥50% of `M4`'s gain.

**H5 (limit of applicability).** `M4`'s gain falls as the variance along `v̂` (`v̂ᵀΣv̂`) grows and as
the tilt `κ_tilt = ‖P⊥Σv̂‖/(v̂ᵀΣv̂)` falls. At `κ_tilt → 0` (`v̂` an eigenvector of `Σ`) `M4`
degenerates algebraically into `M1` with a retuned strength, and there should be no gain. Prediction:
the rank correlation of the gain with `κ_tilt` is positive, |ρ_s| > 0.5 on the 12 TEST features.

**H6 (strong method).** A repair trained against a frozen downstream (`M6`, §5.4) will beat
`M4`, because it optimizes what we care about (not breaking the computation of the upper layers), and not a
proxy MSE at a single layer. Prediction: `M6` > `M4` on the primary endpoint, but by no more than a factor of 1.5.

All six are falsifiable. A negative outcome for any of them is a result, not a failure; §6 shows
exactly how it would be visible.

---

## 2. Setup

### 2.1 Model and intervention point

- **GPT-2 small** (124M, 12 layers, d=768), TransformerLens.
- Hook `blocks.6.hook_resid_post` — "after the middle layer". Verified empirically:
  `blocks.6.hook_resid_post` and `blocks.7.hook_resid_pre` are bit-for-bit the same tensor
  (`max|Δ| = 0.0`), so the SAE dictionary of the `gpt2-small-res-jb` release (trained on `hook_resid_pre`)
  lives exactly in the basis of the intervention.
- The intervention is applied to all prompt positions and to every new token. With a KV cache this
  means: on the first pass the hook sees all prompt positions, afterwards only the new token. This is
  the correct implementation of "at all positions"; in the code it is one and the same hook function.
- Positions 0–1 are excluded from the estimation of `μ, Σ, median‖h‖` and from the intervention: in GPT-2 the residual
  norm at the first positions is an outlier (attention sink), which contaminates the statistics and the `α` scale.

### 2.2 Directions and strength parameterization

- `v = W_dec[f]` (a column of the SAE decoder), `v̂ = v/‖v‖`.
- Strength is set by the increment of the concept coordinate: `s = c · median‖h‖`, `α = s/‖v‖`. All methods
  are compared at the same `s`, and `M5` is arranged so that `v̂ᵀδ = s` holds by construction.
- Grid `c ∈ {0, 0.5, 1.0, 1.5, 2.0, 3.0}`. The density near the inflection point is refined on DEV
  (the fluency collapse is a threshold effect, not a smooth one).

### 2.3 Feature selection — automatic, not by hand

Hand-picking 6 "nice" features only allows one to claim 6 case studies. Hence selection by an objective
rule on corpus activations:

1. For each feature, the top-activating tokens are collected on a held-out slice of the corpus.
2. Lexical coherence metric: the share of activation mass falling on the 10 most frequent
   tokens among the top-200 activations. A feature passes if the share > threshold and the activation frequency is in a
   reasonable corridor (not dead, not omnivorous).
3. From those that pass, **12 TEST features** and **6 DEV features** are drawn at random (fixed seed).
4. Each feature's token list automatically becomes the keyword set for the model-free
   concept metric; a one-phrase human-readable description is composed from these tokens and goes
   into the report and into the judge prompt.

This removes the question "did you pick the features to suit the method" and makes the keyword hit rate reproducible.

---

## 3. Splits and the freezing rule (R4 + protection against test-time leakage)

Three disjoint sets of feature indices:

| Set | Size | Role |
|---|---|---|
| `FIT` | the rest of the dictionary | synthesis of training perturbations only |
| `DEV` | 6 | choice of all knobs: architecture, σ/β, λ, η, judge prompt, the `c` grid |
| `TEST` | 12 | opened once, for the final fronts |

Buffer: any feature with `|cos| > 0.3` to any direction from `DEV ∪ TEST` is excluded from `FIT`.
The report includes a table of `max |cos(v_TEST, u_FIT)|`, so that leakage can be checked rather than taken on trust.

The prompts are split as well: `prompt-DEV` (10) and `prompt-TEST` (30), from different corpus documents; the
activation corpus is a third, disjoint shard.

**Freezing rule.** All inference hyperparameters (knobs) (`λ`, `η`, the `σ` of the analytic denoiser, temperature,
top-p, length, judge prompt) are fixed on DEV and written to `configs/frozen.yaml` with the hash of the
commit **before** the first run on TEST. On TEST only `c` changes. This is essential: a sweep over `λ`
includes `λ=0`, that is, the baseline itself, so a front built as a union over `(c, λ)` would
dominate the baseline tautologically.

---

## 4. Metrics

### 4.1 The x axis — fluency

1. **Primary: log-PPL of the continuation**, computed over the generated tokens only, conditioned on the
   prompt, under **Pythia-410m** — a family different both from GPT-2 (the object) and from Qwen (the judge). GPT-2's own
   perplexity on its own generations is degenerate: a broken model is confident in its own garbage.
2. **Robustness**: the same measurement with a second scorer (Qwen2.5-0.5B base). If the ranking of methods
   changes, that goes into the report.
3. **dist-1/2/3** following Li et al. 2016: unique n-grams, counted at the corpus level and
   normalized by the total number of tokens, at a fixed generation length. Caveat:
   dist-n is non-monotonic — collapse into repetition lowers it, degradation into gibberish raises it; we show
   this on the data.
4. **Repetition rate** (the share of repeated 4-grams) — catches the second failure mode.

### 4.2 The y axis — concept

1. **Primary (model-free): keyword hit rate** over the feature's automatically derived token set,
   excluding tokens already present in the prompt.
2. **SAE feature activation**: the mean activation of the target feature when the generated text is run
   back through the model + SAE. The circularity (the same SAE that `v` was taken from) is acknowledged explicitly; the metric
   is secondary.
3. **LLM judge**: Qwen2.5-1.5B-Instruct, prompt from `persona_vectors`, the score taken as a
   **logprob-weighted expectation** over the valid numbers at `max_tokens=1` — this is more robust than
   parsing free text, and it removes the problem of a noisy scale in a small model.
4. **Guard metric: prompt relevance** (cosine of the embeddings of the prompt and the continuation). Needed because
   at large `α` the model may simply ignore the prompt and still score on the concept metric.

### 4.3 Metric validation (before the main runs)

- Controls: a random direction of the same norm, `−v`, another TEST feature, shuffled concept labels
  for the judge. The concept metric must distinguish these from the true `v`.
- Blind manual annotation of a stratified subsample (~150 texts, blind to method and `α`):
  "text is fluent / broken" and "concept present / absent". Agreement with the automatic
  metrics is computed. This is thirty minutes of work, but it settles the main question about both axes.

### 4.4 Primary endpoint (declared in advance)

**Keyword hit rate at matched fluency**: the value of the concept metric interpolated to a
fixed log-PPL budget equal to the log-PPL of naive steering `M1` at `c = 1.0`. Aggregation — the
mean over the 12 TEST features; the confidence interval — a **paired hierarchical bootstrap** with
resampling at three levels (feature → prompt → sample) and recomputation of the front in each replication.
The remaining metrics are secondary; they serve to check consistency, not to pick a winner.

---

## 5. Methods

Notation: `x = h + αv`, `P = v̂v̂ᵀ`, `P⊥ = I − P`.

### 5.1 Two defects of the task's literal formulation

**Defect A: conflation of two errors.** `D(x) − x` contains both the "steering repair" and the ordinary
reconstruction error `D(h) − h`, which has nothing to do with steering. As a consequence, at `α = 0` the method
`D(h)` already damages the clean activation, and the zero point of the front depends on which `v` we chose.

**Defect B: incorrect measurement of erasure.** The quantity `−⟨D(x)−x, v̂⟩/α` is not defined at `α=0` and
contaminated by defect A. The correct paired estimate of transmission:

```
τ(α) = ⟨D(h + αv) − D(h), v⟩ / (α‖v‖²),      erasure = 1 − τ
```

Both defects are cured by the same operation: subtracting the baseline `D(h)`.

### 5.2 M4 — CDS: Contrastive Direction-preserving Steering-repair

```
Δ_steer = D(h + αv) − D(h) − αv          # the denoiser's pure response to steering
h̃       = h + αv + λ · P⊥ · Δ_steer      # λ is fixed on DEV
```

Properties that make the construction defensible:
- at `α = 0` we have `h̃ = h` **identically**, for any `λ` and any `v` — defect A is eliminated;
- the concept coordinate `v̂ᵀh̃` coincides with naive steering by construction, so the comparison of
  `M4` against `M1` is made at equal concept strength in this coordinate;
- inference cost is two denoiser passes per token (`D(x)` and `D(h)`, the second cached by
  position) plus one dot product. There are no ODE steps, unlike GLP.

**Degeneracy condition, checkable before the runs.** In the linear case, if `v̂` is an eigenvector of
`Σ`, then `P⊥A v̂ = 0`, and `M4` algebraically coincides with `M1` at a retuned strength — there can
be no gain. Therefore `κ_tilt = ‖P⊥Σv̂‖/(v̂ᵀΣv̂)` is computed for all TEST features before generation (H5).

**Factorial diagnostic grid on DEV** (instead of partial reinjection and projection onto
`span{v,x}`, both dropped from the plan): `h̃ = x + λΔ⊥ + γΔ∥`, `λ, γ ∈ [0,1]`. It answers the
review's question "does `Δ∥` matter for fluency": if the optimum has `γ > 0`, pure orthogonalization
is contraindicated, and this has to be known before TEST, not after.

**Is orthogonality to a single `v̂` sufficient.** Not in the general case: preserving a
differentiable concept score requires `∇C(h)ᵀδ = 0`, which coincides with `v̂ᵀδ = 0` only when
`∇C ∥ v̂`. In an SAE the decoder vector `W_dec[f]` and the encoder functional `W_enc[f]` are not
collinear. The ablation therefore includes a variant with projection onto
`span{W_dec[f], W_enc[f]}`, and it is checked directly whether the concept scores of `M4` and `M1`
agree at equal `c`.

### 5.3 M5 — MTS: Mahalanobis transport of steering (requires no training)

The shift that minimally violates the distribution for a given increment of the concept coordinate
is the solution of `min δᵀΣ⁻¹δ` subject to `v̂ᵀδ = s`:

```
δ = s · Σ_γ v̂ / (v̂ᵀ Σ_γ v̂),     Σ_γ = (1−γ)Σ̂ + γ · (tr Σ̂/d) · I
```

Shrinkage `γ` is mandatory: the tail of the spectrum of `Σ̂` estimated over 2M tokens is noisy. `γ`
is chosen on DEV. The concept coordinate coincides with `M1` exactly, by construction. This is at
once a method and the limit of `M4` at large `λ`, which ties the whole line of work into a single
theory: the task's hint → Wiener denoiser → CDS → Mahalanobis-optimal steering.

### 5.4 M6 — CTR: repair trained against a frozen downstream

A proxy MSE at a single layer is not what matters. What matters is: (a) the upper layers are not
broken, (b) the concept is delivered. Both are differentiable, because the upper half of GPT-2 small
is cheap.

For a clean activation `h`, a direction `u ∈ FIT`, a small safe strength `ε` and a working strength
`β`, the frozen suffix of layers 7–11, `F(·)` → final logits, is run:

```
z₀ = F(h),   z_ε = F(h + εu),   z_β = F(h + βu)
g  = (z_ε − z₀)/ε                                  # local causal response of the concept
a  = clip( ⟨z_β − z₀, g⟩ / (‖g‖² + δ) )
z* = z₀ + a·g                                      # teacher: keep what is aligned with g
```

The meaning: the part of the large response that is aligned with *small and still fluent* steering
is declared useful; the nonlinear off-tangent part is declared damage.

A single residual MLP is trained; `u` and the normalized `β` are fed in through a low-rank embedding:

```
h̃ = h + βu + R_θ(h + βu, u, β)
L  = T²·KL( softmax(z*/T) ‖ softmax(F(h̃)/T) ) + γ‖R_θ‖² + ξ‖R_θ(h, u, 0)‖²
```

The last term is an explicit identity control at zero strength. GPT-2 is frozen, the gradient flows
only into `R_θ`. The `DEV ∪ TEST` directions take no part in training.

Cost per step: three half forward passes + one backward pass over six blocks. At batch 8×128 this is
~2 GB VRAM and fits into one night on a 1660 Ti. If the full KL over the vocabulary turns out to be
heavy — top-k logits plus a Huber loss on the final residual.

### 5.5 M3′ — the "do not change the model's structure" variant (the task's hint)

The task's hint about "fine-tuning the existing MLPs" is implemented **on block 7, not 6**: the
block-6 MLP is computed *before* the intervention point and by construction cannot repair an `αv`
that has not been added yet. LoRA rank 8 on `blocks.7.mlp`, trained on the same loss, one
configuration, no sweep. It is positioned as an answer to the hint, not as a competing method.

### 5.6 The denoiser and the training noise — cut to the minimum

- Architectures: **linear** and **residual MLP 768→1536→768** (pre-LayerNorm, GELU, predicts the
  residual). The choice is made on DEV. FiLM conditioning and deeper variants are dropped from the plan.
- Noise: `x = h + β·u`, where `u` is either isotropic Gaussian or a random sparse combination of
  columns of `W_dec[FIT]`; `β` covers the working range of `‖αv‖`, including large values.
  The argument about "a direction shared across the whole sequence" is **dropped**: a token-wise MLP
  does not see neighboring positions, so for it this is statistically indistinguishable from
  independent directions.
- The `t·h + (1−t)ε` scheme from the task's hint is implemented as training directly on `x = h + s·u`
  with the known normalized `s` passed in. Trying to map `t` from a ratio of norms at inference is a
  train/inference mismatch: the signal scale, the mean and the SNR do not match.
- The analytic Wiener denoiser `D(x) = μ + Σ(Σ+σ²I)⁻¹(x−μ)` remains, with its scope stated
  precisely: it is the posterior mean only for Gaussian `h` and isotropic noise of fixed variance;
  for rank-1 steering it is not optimal and serves as a linear baseline and a tool of the theory,
  not as evidence for a mechanism.

### 5.7 Full list of arms and controls

| ID | Method | Role |
|---|---|---|
| M0 | `h` | anchor |
| M1 | `h + αv` | the front to beat |
| M2 | `‖h‖·x/‖x‖` | skeptical training-free baseline |
| M3 | `x + η(D(x) − x)`, `η` from DEV | the task's literal formulation, in its best DEV configuration |
| M3′ | LoRA on `blocks.7.mlp` | answer to the task's hint |
| M4 | **CDS** | the main cheap method |
| M5 | **MTS** | closed form, the limit of the theory |
| M6 | **CTR** | strong method, trained against the downstream |

Controls: `D(h)` (identity at `α=0`), `x+Δ∥`, `x+Δ⊥`, a random direction, `−v`, a different feature,
shuffled judge labels, `D(h)+αv` (denoise first, then steer), `D(h+βv)` with `β` tuned to match the
concept score, DPD with a random projector instead of `v`. Two training seeds for the `M4` denoiser
and for `M6`.

Decision rule: a method is better **only under Pareto dominance** and with non-overlapping bootstrap
CIs on the primary endpoint.

---

## 6. Analysis (the second half of the work)

1. **Steering transmission** `τ(α)` with CIs, for `M3` and `M4`, and a check against the formula
   `τ = 1 − σ²v̂ᵀ(Σ+σ²I)⁻¹v̂` on the analytic denoiser, where it is a theorem. → tests H1, H2.
2. **Factorial grid `(λ, γ)`** on DEV: separates "repair" and "concept strength" causally. →
   justification for choosing pure orthogonalization or rejecting it.
3. **Spectral decomposition**: the energy of the perturbation along the eigen-directions of `Σ`
   before and after repair, for each method. → tests H4, explains the mechanism.
4. **Causal A/C decomposition**: `A = ⟨z_m − z₀, g⟩/‖g‖²` (the useful response that is preserved) and
   `C = ‖z_m − z₀ − A·g‖/(|A|‖g‖)` (nonlinear damage) for all methods. The claim that unifies the
   whole work: at matched `A`, a method wins exactly to the extent that it reduces `C`; this is
   checked by correlating `ΔC` with `Δ log-PPL`.
5. **Predictors of the gain**: rank correlation of the per-feature gain with `v̂ᵀΣv̂` and with
   `κ_tilt`. → tests H5, and this is the only way to tell "the method works" from "we got lucky on
   these features".
6. **Failure analysis**: features where `M4`/`M6` do not win, explained through item 5, plus
   qualitative text examples.
7. **Latency** (tokens/s) of all arms + parameter counts. Cheapness is part of the claim.

Dropped from the plan as low-value per unit of time: the hypervolume of the front, kNN distance over
2M points, layer-by-layer tracing of all layers 7–11 (the spectrum and two downstream checkpoints
are enough).

---

## 7. Order of work

- **E0** ✅ Infrastructure: venv (torch 2.13+cu126, sm_75 verified), GPT-2 + SAE loaded,
  hook identity verified, dump of 2M activations, `μ, Σ, median‖h‖`, prompts.
- **E1** Feature selection by an automatic rule, the `FIT/DEV/TEST` split, the leakage table,
  `κ_tilt` and `v̂ᵀΣv̂` for the TEST features (recorded as predictions). A throughput benchmark of
  generation / PPL / judge on this GPU — the grid is fixed from measured throughput, not from an estimate.
- **E2** Metric validation: the controls of §4.3, blind manual annotation, correlations. Fixing
  `M1`, `M2` on TEST. Freezing the baselines.
- **E3** Training: denoisers (linear, MLP; Gaussian, sparse dictionary) on `FIT`, choice of
  inference hyperparameters (knobs) on DEV, the factorial grid `(λ, γ)`, `η`, `γ_shrink`. Training
  `M6`. Writing `configs/frozen.yaml`.
- **E4** A single run on TEST: all arms, fronts, primary endpoint, bootstrap.
- **E5** The analysis of §6.
- **E6** Report, HTML, the checkpoint artifact and the model card.

Between E2 and E4 the baseline result files are not overwritten.

---

## 8. Artifacts

```
01-mech-interp/
  README.md            installation and exact reproduction commands
  REPORT.md            report
  report.html          reading version
  PLAN.md              this file (preregistration)
  reviews/             external reviews of the plan
  kb/                  knowledge base: 11 notes + 10 primary-source PDFs
  configs/             features.yaml, frozen.yaml, grids.yaml
  src/                 common, activations, features, steering, denoiser, noise,
                       train_denoiser, train_ctr, generate, metrics, analysis
  scripts/             run_e0..run_e6
  results/             csv + png
  artifacts/           checkpoint + model card + push_to_hf.py
```

The checkpoint is prepared in full (weights, card, publication script). Publication to a public
repository is an external action, so it is performed by the account owner with a single command; in
the README it is a separate step.

## 8a. v2.1 revisions after the external reviews of the plan v1 (reviews/)

**Gate on noise energy (otherwise H1 cannot be proven).** The outcome of the H1 test is governed by
the choice of `σ`: if `σ√d ≪ ‖s·v̂‖`, the denoiser's input lies outside the training ball and
"erasure" is an artifact of MLP extrapolation rather than a property of the posterior mean; if
`σ√d ≳ ‖s·v̂‖`, there is almost no erasure. Therefore, in all arms the perturbation energy is sampled
**from one and the same distribution** covering the deployment range, and only the structure of the
direction changes. The arm with fixed `σ` remains as an ablation that demonstrates the artifact
itself: the `τ(α)` plot at three different `σ_train` values is direct evidence that the
extrapolation question has been asked and closed.

**Judge validity gate (before the main runs).** The judge metric is admitted for use only if, for
each feature, the AUC separating `M0` from `M1` at maximum `c` exceeds 0.9, and the rank correlation
of the judge with manual annotation of ~100 texts is significant. Features that fail the gate are
carried with the keyword hit rate and the SAE metrics only, with an explicit caveat in the report.
The primary endpoint is built on the model-free keyword hit rate anyway, so a failed gate does not
break the work.

**M7 — FSR: Feature-Surgery Repair (requires no training, interpretable by construction).**
The layer-7 SAE is already in the stack, zero new components:

```
z  = enc(h + s·v̂)
z'_j = μ_j + softclamp(z_j − μ_j, k·σ_j)   for all j outside {target ∪ buffer}; target is untouched
h̃  = dec(z') + (x − dec(z))               # error passthrough: damage@s=0 is identically zero
```

`μ_j, σ_j` are the activation statistics of feature `j` on the same corpus dump; `k ≈ 3…4` is chosen
on DEV. Hypothesis: the breakdown of fluency is mediated by a small set of "captured" non-target
features rather than by a diffuse shift. The clamp returns the input of layers 7–11 into its natural
envelope while preserving the drive of the target feature. Falsifiable consequence: the gain is
proportional to the number of anomalous non-target features.

Why this matters more than the geometric arms for this track: the list of the most clampable
features is a **named breakdown mechanism** at the level of interpretable features, not a spectral
hint. The "why and how it works" section gets feature-level evidence.

**Kill switch before FSR.** First the roundtrip error `x → enc → dec → x` is measured on clean
activations: if damage@s=0 from the SAE reconstruction is comparable to the breakdown caused by
steering, the idea is dropped before any investment. The error passthrough in the formula above
mostly cures this, but the check is mandatory and costs 15 minutes.

**Distribution-shift check on rollouts.** The denoiser is trained on corpus activations, while at
inference it sees activations generated from an already shifted context. The divergence (norm,
Mahalanobis) between corpus activations and rollout activations is measured; if the divergence is
substantial, a mixture of corpus + generations is added to the training set. This is a known bias
and it is recorded in the report regardless of the outcome.

**Implementation priority (protection against overscoping).** Mandatory minimum: `M0–M3`, `M4` (CDS),
`M5` (MTS), `M7` (FSR) — all cheap, none of them requires a night. `M6` (CTR) and `M3′` (LoRA) are
the stretch goal, done only if a night remains after a full run of the mandatory minimum.

The final narrative line of the report: diagnosis of transmission `τ` → cheap geometric repair (CDS)
→ its closed form (MTS) → the question "what exactly is broken, not where" (FSR, a feature-level
mechanism) → an upper bound from training against the downstream (CTR).

## 9. Budget

The grid is fixed after the benchmark in E1. Reference point: 12 features × 6 values of `c` × 30
prompts × 1 sample × 8 arms ≈ 17k generations of 48 tokens. Batched generation with GPT-2 small
takes minutes; the 410M PPL scorer, tens of minutes; the 1.5B judge is the heaviest part — if time
runs short, judging is done on a stratified subsample of 3–5k texts with CIs, while the full grid is
scored with the two cheap concept metrics. Training the denoiser takes minutes, `M6` one night. Two
nights in total, with margin.
