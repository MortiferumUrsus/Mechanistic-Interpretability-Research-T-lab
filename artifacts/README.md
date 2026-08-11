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

# Residual-stream denoiser for repairing activation steering in GPT-2 small

A small residual MLP that takes an activation from the residual stream of GPT-2 small after block 6
and returns an estimate of the unperturbed activation. Its purpose is to repair the fluency damage
caused by large-coefficient activation steering along a sparse-autoencoder decoder direction, while
leaving the steering signal itself intact.

Trained for the Mechanistic Interpretability track of the T-Lab 2026 selection. Code, protocol and
the full report: see the repository linked below.

## What it is for

Steering adds a direction to the residual stream, `h + s·v̂`, where `v̂` is a normalised decoder
column of an SAE. Large `s` delivers the concept but breaks the text. The intended use of this
checkpoint is **not** `denoiser(h + s·v̂)` directly: applied that way it removes a large part of the
steering signal along with the damage. It is meant to be used contrastively, keeping the component
along the steering direction untouched:

```python
d_steer = D(h + s * v_hat, s) - D(h, 0.0) - s * v_hat
d_perp = d_steer - (d_steer @ v_hat).unsqueeze(-1) * v_hat
h_tilde = h + s * v_hat + lam * d_perp
```

Two properties follow by construction: at `s = 0` the transform is exactly the identity, and the
concept coordinate `v̂ᵀh̃` equals that of naive steering, so comparisons happen at equal concept
strength.

## Interface

- Input: float32 tensor with last dimension 768, taken at `blocks.6.hook_resid_post`
  (TransformerLens naming; bit-identical to `blocks.7.hook_resid_pre`).
- Second input: the magnitude of the perturbation being shown, in activation-norm units. The model
  is conditioned on it; passing zero tells it the input is clean.
- Output: same shape, the estimated clean activation.
- The first two token positions of a sequence are excluded from the intervention in the reference
  implementation: the residual norm there is an outlier and distorts both statistics and repair.

## Training

- Objective `‖h − D(h + s·u)‖²` on residual activations of OpenWebText.
- Perturbation directions `u`: a mixture of isotropic Gaussian directions and random sparse
  combinations of SAE decoder columns, drawn only from a **training** subset of latents.
- Magnitudes `s` log-uniform over the deployment range, identical across all training variants, so
  that ablations over the noise structure are not confounded by noise energy.
- 10% of examples are clean, which keeps the model close to the identity on unperturbed input.

## Leakage control

The SAE latents used to build training perturbations are disjoint from the latents used for
selection and for evaluation. Any latent with absolute cosine similarity 0.3 or above to any
evaluation direction is excluded from the training dictionary; the measured maximum over all pairs
is reported in the repository.

## Limitations

- GPT-2 small only, and only at the one intervention site it was trained for.
- Trained on corpus activations, applied at generation time to activations produced by an already
  steered context; that distribution gap is measured in the report rather than assumed away.
- The closed-form covariance-based alternative described in the report reaches a large share of the
  same benefit with no training at all. This checkpoint is worth its weight only where that share
  is not enough.
