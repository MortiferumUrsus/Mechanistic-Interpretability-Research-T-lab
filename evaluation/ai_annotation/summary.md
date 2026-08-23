# Task 1 generated-text annotation by one AI subagent

## Disclosure

All 468 item labels were produced by **one AI subagent in a separate agent context, not by humans**. The exact runtime backend model identifier was not exposed to the annotator. Human inter-rater reliability and human–AI agreement are unavailable. This is an item-to-arm-blind AI audit, not a human evaluation and not an externally registered study.

The Task 1 README and report were read during source discovery, as requested, before scoring. The annotator therefore knew the stated hypothesis/prior aggregate claims but did not see the randomized item-to-arm mapping or stored automatic scores while assigning the 468 item scores. The target keyword/top-token lists are SAE-derived, so `target_success` is not independent concept ground truth; coherence and prompt relevance are the more external linguistic checks.

## Sampling frame and frozen design

The exact frame was all 7,560 current, non-stale rows in `results/gen_r3.jsonl` (5,760) and `results/gen_r3_ctrl.jsonl` (1,800). A deterministic balanced condition-coverage sample selected **N=468 outputs** across all 12 `test_r3` features: 180 `dirfix`, 180 `naive`, and 108 `randrot`. It contains **180 matched primary-arm feature–strength–prompt cells**. This is not a simple random sample and its unweighted percentages do not estimate the production-frequency mixture of all 7,560 rows.

- Sampling/randomization seed: `task01-ai-annotation-20260823-v1`
- Randomized blind item manifest SHA-256: `e455554d53c1e8949fe1dff6fe53ae80862bcaa4f2fc17f8f9400c63309f1758`
- Randomized left/right display manifest SHA-256: `7b57ad5254ecd2e13b67a83deb00671bacd44f41f9e60731712a94073867feec`
- Original locally frozen rubric SHA-256: `66b98ebde025939b0290d19a0e2c8e8c39ac0058641d514810a36dd0fbd2b0b0`
- Masked aliases after unblinding: `MASK_A=dirfix`, `MASK_B=randrot`, `MASK_C=naive`

The rubric was locally pre-specified and frozen before arm-label inspection and candidate scoring. This timestamp/hash freeze is not an external preregistration. `protocol.md` is preserved byte-for-byte; `protocol_deviations.md` records the pre-unblinding amendment and the hypothesis-blinding clarification.

## Rubric and thresholds

Every item received integer 1–5 scores for coherence/fluency, prompt relevance, target success, degeneration control, safety/integrity, and overall quality, plus defect flags and a short evidence-based rationale. Higher is better. `coherent`, `relevant`, and `concept present` mean a score of 4–5. A degeneration/repetition issue means degeneration control 1–3; severe means 1–2.

## Output-level results

| Arm/design | N | Coherent | Relevant | Concept present | Degeneration/repetition issue | Severe degeneration | Overall mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | 468 | 112/468 (23.9%) | 159/468 (34.0%) | 236/468 (50.4%) | 269/468 (57.5%) | 187/468 (40.0%) | 2.077 |
| dirfix | 180 | 52/180 (28.9%) | 72/180 (40.0%) | 122/180 (67.8%) | 113/180 (62.8%) | 95/180 (52.8%) | 2.167 |
| naive | 180 | 43/180 (23.9%) | 58/180 (32.2%) | 74/180 (41.1%) | 96/180 (53.3%) | 61/180 (33.9%) | 2.078 |
| randrot | 108 | 17/108 (15.7%) | 29/108 (26.9%) | 40/108 (37.0%) | 60/108 (55.6%) | 31/108 (28.7%) | 1.926 |

The `truncated_or_incomplete` flag occurred in 271/468 stored continuations: dirfix 79/180, naive 119/180, and randrot 73/108. Because every stored continuation has a fixed generation budget, this flag often records a budget cutoff/design artifact; it is not interpreted by itself as model failure. The scored degeneration dimension instead emphasizes visible repetition, looping, and collapse, and primary-arm comparisons are matched.

## Matched primary-arm effects

Differences below are paired within the 180 matched cells and use a 20,000-replicate bootstrap clustered by 12 SAE features (seed 20260823).

| Outcome | dirfix − naive mean | Feature-cluster bootstrap 95% CI | Matched cells |
|---|---:|---:|---:|
| Coherence/fluency | +0.083 | [-0.083, +0.239] | 180 |
| Prompt relevance | +0.189 | [-0.011, +0.383] | 180 |
| Target success | +0.978 | [+0.567, +1.378] | 180 |
| Degeneration control (higher is better) | -0.522 | [-0.856, -0.200] | 180 |
| Safety/integrity | +0.056 | [-0.128, +0.239] | 180 |
| Overall quality | +0.089 | [-0.094, +0.244] | 180 |
| Mean of five dimensions | +0.157 | [+0.004, +0.292] | 180 |

The strict result is a target-quality tradeoff. `dirfix` has substantially higher target success, but worse degeneration control. The intervals for coherence, prompt relevance, safety/integrity, and overall quality include zero. The small positive interval for the five-dimension mean combines opposing target and degeneration effects and must not be presented as across-the-board text-quality improvement.

## Paired analysis mechanically derived from blind item scores

No direct pair-annotation pass, second judgment, or pair-specific rationale was performed. The following are deterministic comparisons of the already-saved blind item scores for 180 matched cells: overall compares `overall_quality`; fluency compares the mean of coherence and degeneration control; concept compares `target_success`; exact equality is a tie. The rules were frozen before unblinding, after item scoring.

| Derived comparison | dirfix wins | naive wins | Ties | dirfix share among decisive | Feature-cluster bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| overall | 43 | 24 | 113 | 64.2% | [+0.449, +0.815] |
| fluency | 32 | 56 | 92 | 36.4% | [+0.220, +0.545] |
| concept | 91 | 12 | 77 | 88.3% | [+0.784, +0.947] |

These are derived matched-cell comparisons, not 180 pair judgments. They favor `dirfix` for target concept, favor `naive` for the fluency composite, and do not give a conclusive overall preference because the clustered interval for the decisive overall win share includes 50%.

## Exploratory score associations

| Arm/design | Target success vs. | Spearman rho | Feature-cluster bootstrap 95% CI | N |
|---|---|---:|---:|---:|
| ALL | coherence/fluency | -0.098 | [-0.237, +0.060] | 468 |
| ALL | prompt relevance | -0.111 | [-0.260, +0.058] | 468 |
| ALL | degeneration control | -0.232 | [-0.425, -0.005] | 468 |
| dirfix | coherence/fluency | -0.233 | [-0.445, -0.054] | 180 |
| dirfix | prompt relevance | -0.173 | [-0.382, +0.012] | 180 |
| dirfix | degeneration control | -0.332 | [-0.567, -0.092] | 180 |
| naive | coherence/fluency | -0.002 | [-0.174, +0.174] | 180 |
| naive | prompt relevance | -0.049 | [-0.222, +0.136] | 180 |
| naive | degeneration control | -0.050 | [-0.283, +0.178] | 180 |
| randrot | coherence/fluency | -0.049 | [-0.214, +0.116] | 108 |
| randrot | prompt relevance | -0.205 | [-0.389, +0.006] | 108 |
| randrot | degeneration control | -0.110 | [-0.321, +0.135] | 108 |

These dependent-sample associations are exploratory, not causal. In particular, they do not establish that concept strength causes degradation; they indicate where the single AI judge's scores covary in this condition-balanced sample.

## What this closes—and what it does not

This package closes a real-output, item-to-arm-blind, single-AI text-quality screen for the stated 468-output sample. It supports the claim that `dirfix` more often expresses the SAE-derived target than `naive`. It also supplies direct counterevidence to any unqualified claim that `dirfix` improves human-like text quality: degeneration is worse, while coherence/relevance/overall intervals cross zero.

It does **not** establish human preference, human-readable quality in the population, inter-rater reliability, deterministic reproduction of the AI judgments, independent semantic ground truth, generalization to all 7,560 outputs or other models/features/prompts, causal mechanism validity, or correctness of perplexity/activation/Pareto metrics. The deterministic sample, joins, derived fields, aggregations, and intervals are reproducible from the saved labels; the interactive AI labeling act is not claimed reproducible.

## Reproduce and validate

From the Task 1 repository root after placing this package at `evaluation/ai_annotation`:

```bash
python evaluation/ai_annotation/prepare_blind_manifest.py --repo-root .
python evaluation/ai_annotation/aggregate_annotations.py --annotation-dir evaluation/ai_annotation
python evaluation/ai_annotation/build_public_summary.py --annotation-dir evaluation/ai_annotation
python evaluation/ai_annotation/build_provenance.py --repo-root . --annotation-dir evaluation/ai_annotation
python evaluation/ai_annotation/validate_package.py --repo-root . --annotation-dir evaluation/ai_annotation
```

The last command must print `VALIDATION PASS`. Re-running manifest construction must preserve the recorded manifest hashes. Source and package hashes are in `source_hashes.csv`, `manifest_hashes.json`, and `SHA256SUMS.txt`.
