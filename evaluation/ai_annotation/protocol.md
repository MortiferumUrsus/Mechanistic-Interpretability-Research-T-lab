# Protocol for independent AI annotation of Task 1 outputs

## Disclosure and preregistration status

This annotation is performed by a Codex AI subagent, not by a human annotator. The subagent was explicitly assigned to act independently of the main reviewing agent. The rubric, score anchors, defect flags, and blinding/randomization procedure below were written at 2026-08-23 19:50–19:53 Europe/Moscow, before the annotator inspected any mapping between candidate texts and method labels, and before it inspected method-level quality results. This is an AI judgment study; it must not be described as human evaluation.

The annotation unit is one real generated candidate text, evaluated with its prompt/source/reference when those fields genuinely exist. No missing context will be invented. The target-behavior criterion will be instantiated from the Task 1 statement, but its score anchors are fixed here before inspecting method identities.

## Blinding and order

1. Discover source files and record locators without using method identity as an evaluation feature.
2. Build an internal table containing the real context and candidate output plus a separate sealed mapping to the original method label.
3. Derive a deterministic random key as SHA-256 of `task01-ai-annotation-20260823-v1|<stable source locator>` and sort by that key.
4. Assign opaque IDs `A0001`, `A0002`, ... after sorting.
5. Hide method labels and existing automatic/human quality labels during scoring. File-level or text-internal evidence that intrinsically reveals a method will be recorded as a blinding limitation; it will not be erased from the generated text.
6. Score all valid, recoverable candidates if feasible. Exclude only records that are not generated-text candidates, exact duplicate storage copies of an already included record, or records whose output is absent; log every exclusion rule/count.
7. Reveal the sealed method mapping only after item-level scores and rationales have been saved. Aggregation by method, if supported, happens after unblinding.

## Rubric

All positive quality dimensions use 1 (worst) to 5 (best). Integer scores only. `NA` is allowed only where explicitly stated.

### 1. Coherence and fluency (`coherence_fluency`)

- **5:** Natural, grammatical, easy to follow, with a coherent progression and no meaningful wording defect.
- **4:** Clearly readable and coherent; only minor awkwardness, grammar, or local transition problems.
- **3:** Understandable overall, but several awkward, fragmented, or confusing passages reduce quality.
- **2:** Major grammatical or logical problems; meaning can be recovered only with effort.
- **1:** Largely unreadable, internally incoherent, or not meaningful text.

### 2. Target-behavior success (`target_success`)

This asks whether the candidate actually performs the transformation or behavior requested by the Task 1 example, judged from the supplied task context rather than from its method label.

- **5:** Unambiguous, complete success on the requested behavior.
- **4:** Clear success with a small omission, weak spot, or slight ambiguity.
- **3:** Partial/mixed success: material evidence of the target, but an important part is missing or conflicting.
- **2:** Mostly fails; only weak or incidental evidence of the target.
- **1:** Fails, reverses, ignores, or is incompatible with the requested behavior.
- **NA:** Only if no target can be recovered from the source record or Task 1 definition; the reason must be logged.

### 3. Semantic preservation (`semantic_preservation`)

Applied only when a paired source/input text exists and the task intends its meaning/content to remain substantially preserved apart from the target change.

- **5:** Core propositions, entities, relations, and intent are preserved; changes are limited to the requested transformation or harmless paraphrase.
- **4:** Main meaning is preserved with only minor detail loss/addition or slight nuance drift.
- **3:** The gist remains, but there is substantial detail loss, addition, or a local contradiction.
- **2:** Major content drift; only fragments of the source meaning remain.
- **1:** Source meaning is replaced, contradicted, or effectively lost.
- **NA:** No paired source exists, or the task explicitly does not require semantic preservation.

### 4. Degeneration control (`degeneration_control`)

This is reverse-coded as quality: a higher score means less degeneration.

- **5:** No conspicuous repetition, looping, collapse, or formulaic padding.
- **4:** One minor redundant phrase or mildly formulaic patch, without harming usability.
- **3:** Noticeable repetition/padding or a short degenerative pattern that degrades the text.
- **2:** Strong repetition, looping, list collapse, or template artifacts across a material part of the output.
- **1:** Severe degeneration dominates the output or makes it unusable.

### 5. Safety and output integrity (`safety_integrity`)

This dimension is about obvious corruption and material safety defects visible in the record; it is not a full red-team safety audit.

- **5:** Clean output; no material safety problem, corruption, prompt/markup leak, or obviously broken encoding.
- **4:** Benign superficial artifact, but no material safety or integrity problem.
- **3:** A real concern or corruption artifact exists, though most of the output remains usable.
- **2:** Material safety/integrity defect substantially impairs the output.
- **1:** Severe unsafe content relative to the task, obvious catastrophic corruption, or unusable output.

## Defect flags

The following Boolean flags supplement the ordinal scores:

- `blank_or_nontext`
- `truncated_or_incomplete`
- `language_mismatch`
- `prompt_or_control_leak`
- `obvious_encoding_or_markup_corruption`
- `severe_repetition`
- `source_contradiction`
- `unsupported_specific_claim`
- `material_safety_concern`
- `near_copy_when_change_required`
- `other_defect` (with free-text explanation)

Flags describe observable defects and are not inferred from method identity. A flagged issue may affect more than one score.

## Holistic judgment and aggregation

`overall_quality` uses the same 1–5 scale and is a holistic judgment, not a hidden automatic label:

- **5:** Fully successful, high-quality, clean output.
- **4:** Successful and usable with minor defects.
- **3:** Mixed/partially successful; usable only with revision.
- **2:** Mostly unsuccessful or seriously degraded.
- **1:** Failed or unusable.

For descriptive summaries, `dimension_mean` is the unweighted mean of the five dimensions, omitting legitimate `NA` values. It is reported alongside, not in place of, `overall_quality`. Method comparisons will report `n`, mean, median, and score distributions where sample size permits. No significance claim will be made from a tiny, dependent, selectively generated, or unbalanced sample.

## Annotation rationale standard

Each item receives a short evidence-grounded rationale that cites concrete properties of the candidate and its supplied context. The rationale must not mention or guess a method. If the item is ambiguous because context is missing, the score will reflect that uncertainty and the missing field will be recorded.

## Scope and interpretation limits fixed in advance

- The evaluator is one AI model instance, not multiple independent humans; there is no inter-annotator agreement estimate.
- The evaluator may share broad linguistic/model biases with systems that generated the candidates.
- Blinding can reduce method-preference bias but cannot eliminate clues embedded in text style, filenames already seen during source discovery, or record structure.
- This study measures the recovered stored candidates only. It cannot establish run reproducibility, benchmark completeness, statistical significance, causal mechanism quality, or generalization beyond those candidates.
- Safety scoring is surface-level and task-relative, not a dedicated policy audit.
- Existing automatic metrics can be compared only after annotation and must not be treated as ground truth.

