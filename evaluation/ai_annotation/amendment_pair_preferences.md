# Amendment: pair-preference derivation

Frozen at 2026-08-23T20:15:31+03:00, before any arm mapping was opened and before pair preferences were computed. At this point 420/468 individual blind annotations had been completed.

To avoid inventing a second independent qualitative pass and to make all 180 matched-pair results exactly reproducible, pair preferences will be deterministic projections of the already-blind individual AI scores:

- `overall_preference`: higher `overall_quality`; tie if equal.
- `fluency_preference`: higher sum of `coherence_fluency + degeneration_control`; tie if equal.
- `concept_preference`: higher `target_success`; tie if equal.
- left/right are taken exactly from the precomputed blinded pair manifest.

Thus the pair file is **derived from one AI annotator's item judgments**, not a separate pairwise annotation exercise and not an additional annotator. Pair rationales will report the score differences only. This amendment changes the operational detail in `operationalization.md` that said each pair would receive a fresh qualitative preference/rationale; it does not change the sample, item rubric, thresholds, or individual scores.

## Completion note (2026-08-23T20:33:39+03:00; post-unblinding)

This file is retained as the historical intermediate amendment. The later, pre-unblinding `protocol_deviations.md` is authoritative for what was actually delivered: no direct pair pass and no pair rationales were produced. The package contains only mechanically derived matched-cell preferences from the saved blind item scores.
