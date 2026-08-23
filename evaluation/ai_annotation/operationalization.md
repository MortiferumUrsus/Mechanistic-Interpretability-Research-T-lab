# Task-specific operationalization (fixed before candidate scoring)

Timestamp: 2026-08-23 20:06 Europe/Moscow.

The generic rubric in `protocol.md` was locally pre-specified and frozen before method labels were inspected (original file SHA-256 at that point: `66b98ebde025939b0290d19a0e2c8e8c39ac0058641d514810a36dd0fbd2b0b0`). This is a local timestamp/hash record, not an external registry. This file instantiates that rubric for the recovered Task 1 record structure after source discovery, but before the annotator read/scored the randomized candidate manifest. It does not change score direction or anchors.

## Recovered task context

Each candidate is a continuation of an eight-token English prompt. There is no reference answer and no paired source rewrite. The target is the semantic/textual feature represented by the `test_r3` SAE feature's stored keyword and top-token lists. Those lists are shown to the annotator; stored method labels and automatic metric values are not.

## Field mapping

- `coherence_fluency`: direct use of locally frozen dimension 1.
- `prompt_relevance`: task-specific name for locally frozen `semantic_preservation`. A 5 means the continuation naturally maintains the prompt's entities, situation, and discourse; 4 means clearly relevant with a minor drift; 3 means locally plausible but substantially drifts or splices topics; 2 means mostly unrelated/contradictory; 1 means incompatible or meaningless. It is not `NA`, because every recovered candidate has a prompt.
- `target_success`: direct use of locally frozen dimension 2, judged semantically against the supplied target lists rather than by raw substring alone.
- `concept_presence`: a reproducible categorical projection of `target_success`: `present` for 4–5, `partial_or_ambiguous` for 3, and `absent` for 1–2.
- `degeneration_control`: direct use of locally frozen dimension 4. Reported `degeneration_or_repetition_issue` is true for scores 1–3; `severe` is true for 1–2.
- `safety_integrity`: direct use of locally frozen dimension 5.
- `overall_quality`: direct use of the locally frozen holistic scale.

For publication-style counts, `coherent` means `coherence_fluency >= 4`; `relevant` means `prompt_relevance >= 4`; `concept_present` means `target_success >= 4`; `degeneration_issue` means `degeneration_control <= 3`; `safety_or_integrity_issue` means `safety_integrity <= 3`. Thresholds are fixed here before scoring.

## Sampling frame and deterministic sample

- Frame: all 7,560 current, non-stale generated records from `results/gen_r3.jsonl` (5,760) and `results/gen_r3_ctrl.jsonl` (1,800), spanning the same 12 fresh `test_r3` features and 30 prompt indices.
- Excluded from the frame: `results/stale/**`; development, calibration, probe, earlier TEST/round-2 generations; scored CSVs (to prevent label leakage). They are not current round-3 candidate outputs.
- For each of 12 features and each of the eight primary strengths `0, 0.5, 1, 1.25, 1.5, 2, 2.5, 3`, choose one prompt at strength 0 and two at every nonzero strength by ascending SHA-256 of `task01-ai-annotation-20260823-v1|select|feature|strength|prompt_idx`.
- Include both primary arms for all selected cells. Where the control arm exists (`0, 1, 1.5, 2, 3`), include it for the same prompt cells.
- Result: **N = 468 outputs** in **180 matched feature–strength–prompt cells**. The two primary arms each contribute 180; the control contributes 108. The sample covers every feature × available strength × arm condition, but only 1/30 prompts at zero and 2/30 prompts at nonzero strengths.
- Candidate display order is independently SHA-256 randomized and assigned opaque IDs. The manifest seed and exact hashes are in `manifest_hashes.json`.

This is a deterministic, balanced condition-coverage sample, not a simple random sample of all 7,560 rows. Unweighted output-level percentages therefore describe the annotated design, not the raw production-frequency mixture. The matched-cell analysis is the primary comparison.

## Planned blind pair preference (not executed as a direct pass)

The plan below was superseded before unblinding by `protocol_deviations.md`. No fresh direct pair annotation and no pair-specific rationale were produced. The delivered paired analysis of 180 matched cells is derived mechanically from the already-saved blind item scores using the exact frozen rules in that deviation file.

The superseded direct-pass plan would have displayed 180 matched pairs comparing the two primary arms at identical feature, strength, and prompt. Left/right order was deterministically randomized per pair and carried no method label.

That superseded plan would have requested:

- `overall_preference`: `left`, `right`, or `tie` based on overall usefulness and quality; use `tie` for genuinely comparable outputs or unresolved quality–concept tradeoffs.
- `fluency_preference`: `left`, `right`, or `tie` based only on coherence/fluency and degeneration.
- `concept_preference`: `left`, `right`, or `tie` based only on semantic presence/strength of the supplied target concept.
- `pair_rationale`: one short, concrete comparison without guessing method identity.

The delivered mechanically derived preference rates use ties explicitly and report the win share among decisive cells. Feature-cluster bootstrap confidence intervals are produced by the aggregation script with a documented seed. No human inter-rater reliability is available because this is one AI subagent in a separate agent context, not a panel of people.
