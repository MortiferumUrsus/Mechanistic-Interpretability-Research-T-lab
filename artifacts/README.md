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

# Steering direction correction for GPT-2 small

This repository contains artifacts from a study of SAE feature steering in the residual stream of
GPT-2 small after block six. The main artifact is a learned direction correction

```text
w(v) = normalize(v + A Bᵀv)
```

The matrices have rank 64, for a total of 98,304 parameters. Under the intervention `h + s·w(v̂)`
the norm of the perturbation matches that of the naive variant `h + s·v̂`, so the comparison
reflects a change of direction without a hidden increase in steering strength.

## What it does and what it does not do

On held-out SAE features the correction delivers more of the target feature than the decoder column
at the same perturbation norm: the keyword hit rate of the feature's own vocabulary rises by about
0.3 to 0.4 at matched strength, on the original test set, on twelve features held out of training,
and on 48 fresh features. It also lowers the Pythia log-perplexity of the generated text by about 1.1
to 1.4 nats.

The perplexity gain should not be read as a fluency gain. About half of what the correction adds to
any feature is a single shared direction that lowers the model's next-token entropy; injected on its
own, that direction lowers generation perplexity more than the full correction does, with no concept
delivered, while making the model predict real text worse. A blind audit of generations by a single LLM annotator likewise
confirms the concept gain but finds more degeneration and no improvement in coherence. Measured on
held-out real text at matched concept delivery, the correction reduces the damage of steering,
because it delivers the concept at a lower strength; the size of that reduction (0.07 to 0.6 nats of
teacher-forced negative log-likelihood) is uncertain, since the capability measurement covers only
four features. The full analysis is in the GitHub repository's report, section 10.

`dir_ent.pt` is the same architecture trained with an additional penalty on the change in next-token
entropy. Its mean shared component is smaller by the anatomy statistics (norm share 0.346 against
0.542, mean pairwise cosine 0.119 against 0.294), but the correction cloud is still dominated by one
axis applied with feature-dependent sign (first component 51% of variance against 24%), and its
effect on next-token entropy was not measured. Its real-text comparison at matched concept delivery
is within noise of `direction_correction.pt`, and it delivers less concept per unit of strength.

## Using the correction

```python
import torch
from model import load_direction_correction

correction = load_direction_correction("direction_correction.pt")
v_hat = torch.randn(768)
v_hat = v_hat / v_hat.norm()
w = correction(v_hat.unsqueeze(0))[0]
h_tilde = h + s * w
```

`model.py` contains the module definition and the loading function. In the full implementation the
intervention is applied at all positions except the first two, where the norm of the residual stream
is an outlier that distorts the scale estimate.

The strength is set as `s = c · max_activation · ‖W_dec[f]‖`, that is, relative to the natural scale
of the specific feature. The correction was trained for `c ∈ [0.5, 2.5]`. At `c = 0.5` there is no
improvement; values above `c ≈ 3` were not tested.

## Contents

- `direction_correction.pt` — the main artifact, the rank-64 direction correction (identical to `checkpoints/dir_hot.pt` in the repository)
- `dir_ent.pt` — the entropy-penalized variant
- `denoiser.pt` — the residual stream denoiser from the first round of experiments
- `model.py` — standalone model definitions and loading functions
- `config.json` — architecture and frozen parameters
- `README.md` — this model card (Russian: `README.ru.md`)

`direction_correction.pt` and `dir_ent.pt` take a normalized direction of shape `(..., 768)` and
return a direction of the same shape with unit norm. `denoiser.pt` takes an activation whose last
dimension is 768 and the magnitude of the perturbation.

## Training

The correction was trained through the frozen upper half of GPT-2. For a clean activation `h`, a
direction `v` and a strength `s`, the response of the final logits was optimized:

```text
L = −A / A_naive + γ · relu(C / C_naive − 1)
```

`A` measures the preserved response along a low-signal causal direction, and `C` the relative size
of the nonlinear residual. The reference values are computed in the same batch with the original
direction, so the optimizer cannot obtain a good loss simply by drifting into a safe direction without
delivering the feature. Training took about 4.2 minutes on a GTX 1660 Ti with 6 GB of memory.
`dir_ent.pt` adds `+ 1.0 · mean |H(steered) − H(clean)|`, the change in next-token entropy through
the same frozen suffix.

The denoiser was trained separately on OpenWebText activations with the squared loss
`‖h − D(h + s·u)‖²`. All ablations used the same noise energy distribution.

## Leakage control

The SAE features used for training do not overlap with the features used for parameter selection and
final evaluation. Features with absolute cosine similarity of at least 0.3 to any evaluation
direction were excluded from the training dictionary. The correction was also tested on twelve
features excluded from its training pool before the final retraining, and on 48 further features
excluded, together with their cosine neighbors, from a retrained checkpoint.

## Limitations

- Only GPT-2 small and a single SAE release were tested; a replication at a second layer (block 10)
  finds the same shared direction, with a weaker concept gain.
- The published checkpoint has one training seed. Across five seeds at rank 64 the concept gain is
  stable (0.39 to 0.46) and the perplexity gain varies (−1.1 to −2.1 nats).
- The evaluation rests on twelve to 48 features selected by an objective lexical rule.
- The perplexity gain on generated text is largely a confidence artifact, not improved text quality.
- During generation the activation distribution shifts as the strength grows, while the correction
  remains a fixed linear map.
- Transfer to other models and SAEs was not tested.

The full experimental protocol, tables, plots and mechanism analysis are in the GitHub repository.
