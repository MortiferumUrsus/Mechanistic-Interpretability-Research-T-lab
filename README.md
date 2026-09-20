# Repairing SAE feature steering in GPT-2 small

Steering a language model along an SAE decoder direction, `h̃ = h + αv`, amplifies the feature but
breaks the text at large `α`. This repository studies how to reduce that damage cheaply, in four
rounds: the task's proposal (a denoiser applied after the intervention) and three repairs of it; a
learned correction of the injection direction; a replication on held-out features; and a dissection of
what the learned correction actually does. Report: [`REPORT.md`](REPORT.md) (Russian:
[`REPORT.ru.md`](REPORT.ru.md)), also rendered as a single HTML file, `report.html`. Preregistration:
[`PLAN.md`](PLAN.md). The original assignment: [`TASK.md`](TASK.md) (English translation:
[`TASK.en.md`](TASK.en.md)).

This started as a solution to the T-Lab 2026 test assignment (Mechanistic Interpretability track);
round four was added after the submission.

## Results in four lines

1. **The denoiser erases the signal.** A denoiser trained to pull activations back to the corpus
   manifold removes 24 to 62% of the steering vector, in agreement with a closed-form prediction, and
   every post-hoc repair of the activation increases the nonlinear part of the downstream response.
   No repair beats naive steering (report, sections 2 to 8).
2. **A learned direction correction delivers more concept.** A rank-64 map `w(v) = normalise(v + M v)`
   (98k parameters), trained on one set of features and applied to others at the same perturbation
   norm, raises the preregistered endpoint on TEST from 0.534 to 0.776 and replicates on 12 held-out
   features and on 48 fresh ones (sections 9, 10.8).
3. **Its perplexity gain on generated text is largely an artefact.** Half of the correction is one
   shared direction that lowers the model's next-token entropy. Injected alone it reproduces the
   whole Pythia log-PPL gain at zero concept while making the model predict real text worse.
   Generation perplexity under an external scorer is not a fluency measure once the intervention can
   change the model's confidence (sections 10.2 to 10.6).
4. **On real text at matched concept delivery the learned correction still costs less than naive
   steering**, by 0.1 to 0.6 nats of teacher-forced negative log-likelihood depending on how the two
   curves are joined, with wide intervals: the measurement (four features) establishes the sign, not
   the size. An entropy-penalised variant halves the confidence component and leaves that comparison
   unchanged (sections 10.6, 10.7).

## Setup

GPT-2 small (TransformerLens), intervention at `blocks.6.hook_resid_post`. That tensor is bitwise
identical to `blocks.7.hook_resid_pre`, on which the `gpt2-small-res-jb` SAEs are trained, so decoder
columns are steering directions in the intervention basis without any change of basis
(`activations.py verify` checks the identity).

Strength is parameterised as the increment of the concept coordinate `s = c · natural_s(f)`, where
`natural_s(f) = ceiling_f · ‖W_dec[f]‖` is the feature's own strongest activation on the corpus
(`src/common.py`, `natural_strength`). Natural scales differ by a factor of 6.3 across features
(`natural_s` = 11.12 to 70.60, `results/feature_scales.csv`), which a global unit such as
`median‖h‖ = 88.23` would hide; in that unit the actual strengths are 0.126 to 0.800 of `c`. The
`c` grid is therefore comparable across features, and arms are compared at equal concept strength.

Fluency is proxied by the log-perplexity of the continuation under Pythia-410m (the generating
model's own perplexity is degenerate: a broken model is confident in its own output), plus dist-1/2/3,
the repeated-4-gram fraction and prompt dependence; concept by the hit rate of the feature's
automatically derived keyword list and by the feature's SAE activation on the generated text. Round
four adds a measurement on text the model did not write: teacher-forced negative log-likelihood and
top-1 accuracy on held-out documents under the same steering hook (`src/capability_sweep.py`).

### Arms

| arm | what it injects or does | training |
|---|---|---|
| `clean` | `h` | — |
| `naive` | `h + s·v̂` | — |
| `norm_preserving` | rescales the result to `‖h‖` | — |
| `denoise_naive` | `x + η(D(x) − x)`, the task's literal formulation | denoiser |
| `cds` | `x + λ·P⊥(D(x) − D(h) − s·v̂)`, contrastive direction-preserving correction | denoiser |
| `mts` | minimal-Mahalanobis-norm shift with the same concept-coordinate increment | — |
| `fsr` | clamps non-target SAE latents to their corpus ceiling | — |
| `dirfix` | `h + s·w(v̂)`, learned direction correction at the naive norm | direction correction (round 2) |
| `randrot` | control: rotation of `v̂` by the same angle in a random direction | — |
| `shared` | `h + s·normalise(v̂ + κ·d̄)`, `d̄` = shared component of the learned correction | — (round 4) |
| `shared_only` | control: `h + s·d̄`, no feature direction | — |
| `residual`, `purified`, `centred`, `diffmeans`, `diffmeans_purified`, `antimanifold`, `rotate` | round-4 direction families, see report section 10.1 | — |
| `cond_wiener`, `cond_denoise` | conditional denoisers pulling toward the concept's own manifold (failed, section 10.9) | conditional denoiser |

Round one (`clean` to `fsr`) was declared before any run. `dirfix` is round two, chosen after round
one was seen and declared as such. The correction is trained only on FIT features and applied to
features it never saw; its perturbation norm equals the naive one by construction. Round three is
the same arm on twelve fresh features (`test_r3`) held out of the correction's training, plus the
random-rotation control. Round four is the set of arms below `randrot` in the table, the
entropy-penalised checkpoint `dir_ent`, the real-text capability sweep, seed and rank sweeps, 48
fresh features (`test_r4`) and a replication at layer 10.

## Installation

```
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

The order matters: `transformer-lens` and `sae-lens` may replace the CUDA build of torch with a CPU
build, so the CUDA build is installed last. Rounds one to three ran on a GTX 1660 Ti (6 GB); round
four ran on Kaggle T4/P100 sessions with the pins in `requirements.txt`. No exact lock file was kept,
so this is a working recipe rather than a bit-identical environment.

## Reproduction

Rounds one to three, from the repository root. Each `scripts/*.sh` has a PowerShell twin
`scripts/*.ps1`, which is what the reported runs used; set `PY` to your interpreter if it is not
`python`.

```
scripts/run_all.sh        # data, feature selection, denoisers (60000 steps), one TEST pass, mechanism, rounds 2 and 3
scripts/run_round2.sh     # round two only: train the correction, select on DEV, one TEST pass
scripts/run_round3.sh     # round three only: select the holdout FROM fit, retrain without it, then generate
```

By stage:

```
cd src

# data: hook identity, activation dump, prompts, SAE latent ceilings
python activations.py verify
python activations.py dump --n-tokens 2000000 --batch-size 16
python activations.py prompts --n-prompts 40 --prompt-len 8
python sae_stats.py --chunk 2048

# feature selection and the FIT / DEV / TEST split
python features.py select --seed 0

# denoisers, trained on FIT directions only; the frozen variant is the first one
python train_denoiser.py --arch mlp --noise gauss --cond 1 --seed 0 --steps 60000
python train_denoiser.py --arch mlp --noise mix --cond 1 --seed 0 --steps 60000
python train_denoiser.py --arch linear --noise mix --cond 0 --seed 0 --steps 60000

# every inference hyperparameter is chosen on DEV, without generation
python sweep_dev.py

# one pass on TEST with frozen hyperparameters
python generate.py --split test --out gen_test.jsonl
python metrics.py --gen gen_test.jsonl --out scored_test.csv --stages ppl,keyword,sae,dist
python pareto.py --scored scored_test.csv --concept keyword_hit

# mechanism
python analysis.py transmission
python analysis.py spectral
python analysis.py surgery
python analysis.py causal --lam 1.5 --shrink 0.01
python analysis.py predictors
python dirfix_vs_naive.py       # A/C and perplexity in one table, cited by report section 9.1
python report_numbers_check.py  # every number in the report must be in the file its paragraph cites
```

The order inside round three matters: `select_round3.py` first reserves twelve holdout features from
FIT, `train_direction.py` then retrains the correction without them (23988 directions instead of
24000), and only then come generation, metrics and the random-rotation control. Swapping the steps
would train the correction on its own holdout.

**Round four** was run on Kaggle through the drivers in `scripts/run_exp*.py` and
`scripts/run_layer.py` (stages selectable with `--stages`; each stage is wrapped so that one failure
does not stop the queue, which also means a green queue summary is not proof that a stage produced its
artefact). The Kaggle scaffolding is in `kaggle/` (see `kaggle/SMOKE.md`). The same stages run locally
from `src/`:

```
cd src
python anatomy.py --ckpt dir_hot                # shared direction d̄ -> checkpoints/shared_direction.pt, anatomy tables
python direction_report.py                      # unembedding diagnostics per direction family
python qq_test.py --n-docs 500 --ctx 512 --c 1.0 --features 3 --split test_r3 --out-prefix qq
python generate.py --split test_r3 --arms naive,dirfix,shared,shared_only,residual,antimanifold --out gen_expA_r3.jsonl
python generate.py --split test_r3 --arms naive,dirfix,shared,diffmeans,centred,purified,diffmeans_purified,rotate --out gen_expF_r3.jsonl
python capability_sweep.py                      # real-text NLL and top-1 under the steering hook
python matched_concept_capability.py            # real-text damage at matched concept delivery
python matched_repetition.py --scored scored_expF_r3.csv --out-prefix matchedF
python feature_distribution.py --scored scored_expF_r3.csv --out-prefix featdistF
python layer_profile.py                         # per-layer norm, projection on d̄, final entropy
python train_direction_ent.py train --rank 64 --steps 2000 --name dir_ent --ent-weight 1.0
python anatomy.py --ckpt dir_ent --tag _ent
python select_round4.py --seed 4242 --n 48      # writes configs/features_r4.yaml; features.yaml is frozen
python train_direction.py train --rank 64 --gamma 1.0 --lr 3e-3 --steps 2000 --name dir_r4 --exclude-r4
```

`TLAB_LAYER=10` moves the whole pipeline to the SAE at `blocks.10.hook_resid_pre`
(`scripts/run_layer.py --layer 10` does the full chain on a throwaway checkout).

Checks that do not need a GPU: `python test_arms.py` and `python test_pareto.py` (invariants of the
arms and of the front machinery), `python report_numbers_check.py` (every decimal in the report is in
the artefact its paragraph cites), and the annotation-package validator below.

## Leakage control

SAE feature indices are split into three disjoint sets. `FIT` (24000 features) is used only to
synthesise training perturbations, `DEV` (6) for every hyperparameter, `TEST` (12) is opened once.
Any feature with `|cos| ≥ 0.3` to any `DEV ∪ TEST` direction is excluded from `FIT`; the actual
maximum over all pairs is in [`results/leakage.csv`](results/leakage.csv). The twelve round-three
features are additionally excluded from `FIT` and the correction retrained without them; the 48
round-four features and every FIT direction within `|cos| ≥ 0.3` of them are excluded from the
checkpoint used in round four (`dir_r4`). Evaluation prompts come from a different corpus shard than
the training activations.

One weakness is stated explicitly. DEV and TEST prompts are not separated although `PLAN.md`
required it: `sweep_dev.py` takes the first ten prompts of `data/prompts.json` and `generate.py` the
first thirty, so every DEV prompt is also a TEST prompt, and a third of the TEST prompts were seen
while choosing the inference hyperparameters. Round four used a separate prompt pool for the
48-feature run; its per-row scored table was not archived, so that separation is recorded in the
kernel logs only (report, section 10.8).

All inference hyperparameters (`λ`, `η`, `k`, covariance shrinkage, `σ`) are fixed on DEV and
written to `configs/frozen.yaml` before the first TEST run; only the strength `c` varies on TEST.
This matters because the `λ` sweep contains `λ = 0`, the baseline itself, so a front taken as the
union over `(c, λ)` would dominate the baseline tautologically.

## Post-hoc annotation of saved generations

`metrics.py` has an experimental `judge` stage (Qwen2.5-1.5B-Instruct as an LLM judge). It was never
run and contributes to no reported number. Instead, after the experiments, a single LLM annotator in a
separate context scored 468 saved round-three generations in 180 matched cells, blind to the
arm-to-item mapping, under a rubric frozen before scoring. It is not a human evaluation and has no
second annotator. For `dirfix − naive` the target-success score is `+0.978 [0.567, 1.378]` and the
degeneration-control score `−0.522 [−0.856, −0.200]`; the intervals for coherence and overall quality
cover zero. Labels, protocol and limitations:
[`evaluation/ai_annotation/summary.md`](evaluation/ai_annotation/summary.md). The deterministic parts
are checked from the repository root by

```bash
python evaluation/ai_annotation/validate_package.py --repo-root . --annotation-dir evaluation/ai_annotation
```

The expected final lines are `VALIDATION PASS`, 468 unique labels, 180 matched cells,
`warnings: []`. The three files the validator hashes as critical sources (`configs/features.yaml`,
`results/gen_r3.jsonl`, `results/gen_r3_ctrl.jsonl`) are stored with the exact bytes that were
hashed, CRLF line endings included, and are marked `-text` in `.gitattributes` so that every clone
reproduces them.

## Repository map

```
README.md, README.ru.md         this file, English and Russian
REPORT.md, REPORT.ru.md         the report; report.html / report.ru.html are single-file renderings
PLAN.md, PLAN.ru.md             preregistration (hypotheses, metrics, order of work); the Russian file is the frozen original
TASK.md, TASK.en.md             the assignment as issued (Russian, verbatim, hashed by the annotation package) and its translation
configs/                        features.yaml (frozen splits), features_r4.yaml (round-four features), frozen.yaml (hyperparameters),
                                expA.yaml (shared-arm blend weight), expF.yaml (its sweep grid)
src/                            implementation; see the command tables above
scripts/                        pipelines: *.ps1 as used for the reported runs, *.sh mirrors, run_exp*.py and run_layer.py for round four
kaggle/                         payload builder, kernel generator, push/fetch helpers, smoke test
results/                        csv, json, figures; round-four tables are named after their driver (expA, expF, expG, layer10);
                                results/kaggle_runs/ holds each kernel's queue summary, notes and small outputs
checkpoints/                    denoisers, dir_hot (published), dir_ent, dir_r4, shared_direction*.pt, seeds/ (rank x seed sweep)
artifacts/                      the published checkpoints with a standalone loader, model card (README.md / README.ru.md) and push script
evaluation/                     post-hoc single-annotator audit with item-level labels and a validator
kb/                             literature notes and PDFs
reviews/                        external reviews of the plan v1 (Russian)
```

Round-four result files and what they hold:

| file | content |
|---|---|
| `results/anatomy_summary.json`, `_ent.json`, `_r4.json` | anatomy of `dir_hot`, `dir_ent`, `dir_r4`; per-feature table for `dir_ent` in `anatomy_per_feature_ent.csv` |
| `results/direction_report.csv` | cosine with `d̄`, weak-half energy and logit reach per direction family and feature |
| `results/layer_profile.csv` | per-layer residual norm, projection on `d̄`, final entropy |
| `results/qq_summary.csv`, `qq_tokens_*.csv` | ActAdd per-token diagnostic on real text |
| `results/scored_expA_r3.csv`, `paired_expA_r3.csv` | `shared`, `shared_only`, `residual`, `antimanifold` (κ = 0.75) |
| `results/scored_expF_r3.csv`, `paired_expF_r3.csv` | `shared`, `diffmeans`, `centred`, `purified`, `diffmeans_purified`, `rotate` (κ = 1.0) |
| `results/scored_expF_kappa.csv`, `capability_kappa_*.csv` | the `κ` sweep for `shared` |
| `results/capability_sweep.csv`, `capability_ent.csv` | real-text NLL and top-1 versus strength |
| `results/matched_concept_capability.csv`, `_ent.csv` | real-text damage at matched concept delivery |
| `results/scored_expF_ent_clean.csv`, `paired_expF_ent.csv` | the entropy-penalised correction `dir_ent` |
| `results/matchedA_*.csv`, `matchedF_*.csv`, `featdist*_*.csv` | matched-repetition and per-feature distribution checks |
| `results/scored_layer10.csv`, `paired_layer10.csv` | the layer-10 replication |
| `results/r4_identity_at_zero.csv` | every round-four arm is the identity at `c = 0` |
| `results/kaggle_runs/tlab-mi-a-d/` | seed and rank sweep (`expD_eval_dev.csv`, training logs) and the `dir_hot` anatomy |
| `results/kaggle_runs/tlab-mi-e4-score/` | the 48-feature run: endpoint summary and paired table |
| `results/kaggle_runs/tlab-mi-g-e/` | the conditional-denoiser run |
| `results/kaggle_runs/tlab-mi-layer4/` | the diverged layer-4 run |

## Publishing the checkpoint

`artifacts/` holds `direction_correction.pt` (the round-2 correction, identical to
`checkpoints/dir_hot.pt`), `denoiser.pt`, `config.json`, `model.py` (a standalone loader) and the
model card. Publishing to Hugging Face is a manual step under the owner's account:

```
hf auth login
python artifacts/push_to_hf.py --repo <user>/gpt2-small-steering-direction-correction --dry-run
python artifacts/push_to_hf.py --repo <user>/gpt2-small-steering-direction-correction
```

`--dry-run` prints the file list and the target repository without uploading anything.
