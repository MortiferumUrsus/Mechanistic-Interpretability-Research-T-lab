# Task 1 AI annotation package

This directory contains an item-to-arm-blind annotation of 468 real Task 1 continuations by one AI subagent in a separate agent context, not by humans. The exact runtime backend model identifier was not exposed. Human inter-rater reliability is absent.

Read `summary.md` for results and limitations. `protocol.md` is the original locally frozen rubric; `operationalization.md` maps it to Task 1; `protocol_deviations.md` records the pre-unblinding amendment. The delivered paired analysis is mechanically derived from blind item scores for 180 matched cells; it is not a direct pair-judgment pass.

Machine-readable labels and joins are in `annotations_blind.jsonl`, `annotations.jsonl`, `annotations_with_masked_arm.*`, and `annotations_unblinded.*`. Aggregates and confidence intervals are in the summary CSV files. Sampling manifests, sealed mapping, provenance, hashes, and scripts are included so the deterministic parts can be checked. Interactive AI labels themselves are not claimed deterministic or independently reproducible.

Portable source locators are relative to the Task 1 repository. Exact snapshots of the narrative documents
seen before annotation are included so validation also works from an isolated public clone. Machine-specific
absolute paths are intentionally omitted from the release copy; `provenance_machine_local.json` records that
policy without publishing local paths.
