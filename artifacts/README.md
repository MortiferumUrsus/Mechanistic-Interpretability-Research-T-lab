---
license: mit
base_model: openai-community/gpt2
tags:
  - interpretability
  - activation-steering
  - sparse-autoencoder
  - gpt2
library_name: pytorch
---

# Steering-direction correction for SAE feature steering in GPT-2 small

A shared low-rank correction of SAE steering directions, `w(v) = normalise(v + A Bᵀ v)`, 98k
parameters, for the residual stream of GPT-2 small after block 6. Injecting `h + s·w(v̂)` instead of
`h + s·v̂` keeps the perturbation norm identical while producing more of the target feature's effect and
less damage to the text.

The interesting part is what this implies: **the SAE decoder column is not the most effective injection
direction for reproducing its own feature's downstream effect.** The correction is trained on one set of
latents and transfers to latents it has never seen, so the improvement is a systematic property of the
decoder basis rather than per-feature tuning.

This repository also contains the residual-stream denoiser from the first round of the same study,
which is kept for reproducibility. The report explains why the denoiser route does not work: it erases
24–62% of the steering signal it is supposed to preserve.

Built for the Mechanistic Interpretability track of the T-Lab 2026 selection. Code, protocol and full
report: see the repository linked below.

## How to use it

```python
w = correction(v_hat.unsqueeze(0))[0]     # unit norm by construction
h_tilde = h + s * w                       # same perturbation norm as h + s * v_hat
```

Strength `s` is expressed in units of the latent's own ceiling on real text, `s = c · max_activation ·
‖W_dec[f]‖`, which is what makes one grid comparable across latents whose natural scales differ by a
factor of six. The correction was trained for `c ∈ [0.5, 2.5]`; at `c = 0.5` it is not an improvement,
and above `c ≈ 3` it is untested.

The first two token positions of a sequence stay untouched in the reference implementation: the residual
norm there is an outlier and distorts both statistics and interventions.

## Interface

Two models ship here and they take different inputs. The headline one is the direction correction; the
denoiser is kept for reproducibility of the first round and is not the recommended model.

**`direction_correction.pt` — the direction correction.**

- Input: a unit-norm steering direction, float32, shape `(..., 768)`. This is a *direction*, not an
  activation: typically a row of an SAE decoder, normalised.
- Output: a unit-norm direction of the same shape, `w(v) = normalise(v + A Bᵀ v)`, rank 64, 98k parameters.
- Use: inject `h + s · w(v̂)` where you would have injected `h + s · v̂`. The perturbation norm is unchanged
  by construction, so this is a rotation of the injected direction and not a change of strength.
- The correction is trained for `c ∈ [0.5, 2.5]` in units of the feature's own activation ceiling; at
  `c = 0.5` it is not an improvement and above `c ≈ 3` it is untested.

**`denoiser.pt` — the residual-stream denoiser (first round, superseded).**

- Input: float32 tensor with last dimension 768, taken at `blocks.6.hook_resid_post` (TransformerLens
  naming; bit-identical to `blocks.7.hook_resid_pre`).
- Second input: the magnitude of the perturbation being shown, in activation-norm units. The model is
  conditioned on it; passing zero tells it the input is clean.
- Output: same shape, the estimated clean activation.

In both cases the first two token positions of a sequence are excluded from the intervention in the
reference implementation: the residual norm there is an outlier and distorts both statistics and repair.

## Training

The objective is measured through the frozen upper half of the network rather than in activation space.
For a clean activation `h`, a training direction `v` and a strength `s`, with `A` the component of the
final-logit response along the concept's small-signal causal direction and `C` the relative size of the
nonlinear residue:

```
L = − A / A_naive  +  γ · relu(C / C_naive − 1)
```

Both references are measured in the same batch with the uncorrected direction, which matters: without the
matched reference the objective is minimised by pointing somewhere harmless, and the concept stops being
delivered at all. Rank 64, 2000 steps, one GPU-hour on a 6 GB card.

The denoiser kept alongside was trained separately with `‖h − D(h + s·u)‖²` on OpenWebText activations,
with perturbation magnitudes drawn from one distribution shared across all its ablations so that noise
structure is not confounded with noise energy.

## Leakage control

The SAE latents used to build training perturbations are disjoint from the latents used for
selection and for evaluation. Any latent with absolute cosine similarity 0.3 or above to any
evaluation direction is excluded from the training dictionary; the measured maximum over all pairs
is reported in the repository.

## Limitations

- GPT-2 small only, one intervention site, one SAE release.
- No improvement at low strength (`c ≈ 0.5`), where the trade in response space goes the wrong way.
- Evaluated on 12 latents selected by an objective lexical rule; the effect is a property of this
  decoder basis and does not automatically transfer to other SAEs or layers.
- The activations it sees at generation time drift away from the corpus as strength rises (norm 89 → 156
  between `c = 0` and `c = 2`); the correction is a fixed linear map and does not model that drift.
