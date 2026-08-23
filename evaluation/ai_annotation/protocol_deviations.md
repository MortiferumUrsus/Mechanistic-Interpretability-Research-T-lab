# Protocol deviations and clarifications

Frozen at 2026-08-23T20:17:00+03:00, after all 468 blinded item annotations were saved and validated, but before opening the arm mapping or computing any arm-level result.

## Direct pair pass not performed

`operationalization.md` planned a fresh qualitative pass over 180 displayed pairs, including a pair-specific rationale. That second pass was not performed. It would not have supplied a second independent annotator and risked inconsistent duplicate judgments under the available completion budget.

Instead, the package contains a **paired analysis of 180 matched cells derived mechanically from the already-saved blind item scores**:

- overall: compare `overall_quality`;
- fluency: compare the arithmetic mean of `coherence_fluency` and `degeneration_control`;
- concept: compare `target_success`;
- exact equality is a tie; otherwise the higher score wins;
- left/right positions come unchanged from the precomputed blinded-pair manifest.

The rules were fixed before unblinding. There are no direct pair judgments, no pair-specific qualitative rationales, and no claim of a second annotation pass. The derived file is named `derived_pair_preferences_blind.*` to prevent that misinterpretation.

## Terminology

The rubric was locally pre-specified and frozen with a timestamp and SHA-256 before inspection of the arm mapping. This is not an externally registered preregistration. Public-facing summaries use “locally pre-specified/frozen,” not an unqualified claim of preregistration.

## Fixed-length endings

Stored candidates are fixed-budget continuations. `truncated_or_incomplete` therefore often reflects the generation-length cutoff, not necessarily method-induced degeneration. Its raw prevalence is not interpreted as model failure; arm comparison is matched, and the primary degeneration dimension focuses on repetition/loops/collapse visible within the stored text.

## Independence limits

Target keyword/top-token lists come from the same SAE feature construction used by Task 1. AI `target_success` is therefore not independent concept ground truth, even though the annotator was blinded to arm labels and automatic scores. Coherence and prompt relevance are the more external linguistic checks.

One AI subagent in a separate agent context produced all item labels. It is not a human and not a panel. Human inter-rater reliability and human–AI agreement are absent. The exact runtime backend model identifier was not exposed to this subagent, so no model name is invented.

The Task 1 README and report were read during source discovery, as the assignment requested. Therefore the study is item-to-arm blind but not hypothesis- or prior-result-blind. The original frozen protocol's phrase “before it inspected method-level quality results” must not be used to claim otherwise; this clarification governs the public interpretation while preserving the original protocol file and hash.
