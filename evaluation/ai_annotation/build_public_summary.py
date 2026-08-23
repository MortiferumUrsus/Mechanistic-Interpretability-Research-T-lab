"""Build the publication-facing Markdown summary from machine-readable aggregates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pct(value: str | float) -> str:
    return f"{100 * float(value):.1f}%"


def score(value: str | float) -> str:
    return f"{float(value):+.3f}"


def ci(low: str | float, high: str | float) -> str:
    return f"[{float(low):+.3f}, {float(high):+.3f}]"


def md_method_table(rows: list[dict[str, str]]) -> str:
    order = {"ALL": 0, "dirfix": 1, "naive": 2, "randrot": 3}
    rows = sorted(rows, key=lambda row: order[row["group"]])
    lines = [
        "| Arm/design | N | Coherent | Relevant | Concept present | Degeneration/repetition issue | Severe degeneration | Overall mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        n = int(row["n"])
        lines.append(
            "| {group} | {n} | {coherent_count}/{n} ({coherent_share}) | "
            "{relevant_count}/{n} ({relevant_share}) | "
            "{concept_present_count}/{n} ({concept_present_share}) | "
            "{degeneration_or_repetition_issue_count}/{n} ({degeneration_share}) | "
            "{severe_degeneration_count}/{n} ({severe_share}) | {overall:.3f} |".format(
                group=row["group"],
                n=n,
                coherent_count=row["coherent_count"],
                coherent_share=pct(row["coherent_share"]),
                relevant_count=row["relevant_count"],
                relevant_share=pct(row["relevant_share"]),
                concept_present_count=row["concept_present_count"],
                concept_present_share=pct(row["concept_present_share"]),
                degeneration_or_repetition_issue_count=row[
                    "degeneration_or_repetition_issue_count"
                ],
                degeneration_share=pct(row["degeneration_or_repetition_issue_share"]),
                severe_degeneration_count=row["severe_degeneration_count"],
                severe_share=pct(row["severe_degeneration_share"]),
                overall=float(row["overall_quality_mean"]),
            )
        )
    return "\n".join(lines)


def md_effect_table(rows: list[dict[str, str]]) -> str:
    labels = {
        "coherence_fluency": "Coherence/fluency",
        "prompt_relevance": "Prompt relevance",
        "target_success": "Target success",
        "degeneration_control": "Degeneration control (higher is better)",
        "safety_integrity": "Safety/integrity",
        "overall_quality": "Overall quality",
        "dimension_mean": "Mean of five dimensions",
    }
    lines = [
        "| Outcome | dirfix − naive mean | Feature-cluster bootstrap 95% CI | Matched cells |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {labels[row['outcome']]} | {score(row['mean_difference'])} | "
            f"{ci(row['cluster_bootstrap_ci95_low'], row['cluster_bootstrap_ci95_high'])} | "
            f"{row['matched_cell_n']} |"
        )
    return "\n".join(lines)


def md_preference_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Derived comparison | dirfix wins | naive wins | Ties | dirfix share among decisive | Feature-cluster bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dimension']} | {row['dirfix_wins']} | {row['naive_wins']} | "
            f"{row['ties']} | {pct(row['dirfix_win_share_among_decisive'])} | "
            f"{ci(row['cluster_bootstrap_ci95_low'], row['cluster_bootstrap_ci95_high'])} |"
        )
    return "\n".join(lines)


def md_association_table(rows: list[dict[str, str]]) -> str:
    labels = {
        "coherence_fluency": "coherence/fluency",
        "prompt_relevance": "prompt relevance",
        "degeneration_control": "degeneration control",
    }
    lines = [
        "| Arm/design | Target success vs. | Spearman rho | Feature-cluster bootstrap 95% CI | N |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {labels[row['y']]} | {score(row['spearman_rho'])} | "
            f"{ci(row['feature_cluster_bootstrap_ci95_low'], row['feature_cluster_bootstrap_ci95_high'])} | "
            f"{row['n']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", default=str(HERE))
    args = parser.parse_args()
    root = Path(args.annotation_dir).resolve()

    methods = read_csv(root / "method_summary_unblinded.csv")
    effects = read_csv(root / "paired_effects_unblinded.csv")
    preferences = read_csv(root / "derived_pair_preference_summary_unblinded.csv")
    associations = read_csv(root / "associations.csv")
    annotated = read_jsonl(root / "annotations_unblinded.jsonl")
    manifest_hashes = json.loads((root / "manifest_hashes.json").read_text(encoding="utf-8"))
    prereg = json.loads((root / "preregistration_record.json").read_text(encoding="utf-8"))

    truncated = Counter(row["arm"] for row in annotated if row["truncated_or_incomplete"])
    method_n = {row["group"]: int(row["n"]) for row in methods}
    protocol_hash = hashlib.sha256((root / "protocol.md").read_bytes()).hexdigest()
    if protocol_hash != prereg["protocol_sha256_at_preregistration"]:
        raise RuntimeError("Frozen protocol hash mismatch")

    summary = f"""# Task 1 generated-text annotation by one AI subagent

## Disclosure

All 468 item labels were produced by **one AI subagent in a separate agent context, not by humans**. The exact runtime backend model identifier was not exposed to the annotator. Human inter-rater reliability and human–AI agreement are unavailable. This is an item-to-arm-blind AI audit, not a human evaluation and not an externally registered study.

The Task 1 README and report were read during source discovery, as requested, before scoring. The annotator therefore knew the stated hypothesis/prior aggregate claims but did not see the randomized item-to-arm mapping or stored automatic scores while assigning the 468 item scores. The target keyword/top-token lists are SAE-derived, so `target_success` is not independent concept ground truth; coherence and prompt relevance are the more external linguistic checks.

## Sampling frame and frozen design

The exact frame was all {manifest_hashes['sampling_frame_n']:,} current, non-stale rows in `results/gen_r3.jsonl` (5,760) and `results/gen_r3_ctrl.jsonl` (1,800). A deterministic balanced condition-coverage sample selected **N={manifest_hashes['annotated_output_n']} outputs** across all 12 `test_r3` features: 180 `dirfix`, 180 `naive`, and 108 `randrot`. It contains **{manifest_hashes['matched_primary_cell_n']} matched primary-arm feature–strength–prompt cells**. This is not a simple random sample and its unweighted percentages do not estimate the production-frequency mixture of all 7,560 rows.

- Sampling/randomization seed: `task01-ai-annotation-20260823-v1`
- Randomized blind item manifest SHA-256: `{manifest_hashes['blinded_manifest_sha256']}`
- Randomized left/right display manifest SHA-256: `{manifest_hashes['blinded_pairs_sha256']}`
- Original locally frozen rubric SHA-256: `{protocol_hash}`
- Masked aliases after unblinding: `MASK_A=dirfix`, `MASK_B=randrot`, `MASK_C=naive`

The rubric was locally pre-specified and frozen before arm-label inspection and candidate scoring. This timestamp/hash freeze is not an external preregistration. `protocol.md` is preserved byte-for-byte; `protocol_deviations.md` records the pre-unblinding amendment and the hypothesis-blinding clarification.

## Rubric and thresholds

Every item received integer 1–5 scores for coherence/fluency, prompt relevance, target success, degeneration control, safety/integrity, and overall quality, plus defect flags and a short evidence-based rationale. Higher is better. `coherent`, `relevant`, and `concept present` mean a score of 4–5. A degeneration/repetition issue means degeneration control 1–3; severe means 1–2.

## Output-level results

{md_method_table(methods)}

The `truncated_or_incomplete` flag occurred in {sum(truncated.values())}/468 stored continuations: dirfix {truncated['dirfix']}/{method_n['dirfix']}, naive {truncated['naive']}/{method_n['naive']}, and randrot {truncated['randrot']}/{method_n['randrot']}. Because every stored continuation has a fixed generation budget, this flag often records a budget cutoff/design artifact; it is not interpreted by itself as model failure. The scored degeneration dimension instead emphasizes visible repetition, looping, and collapse, and primary-arm comparisons are matched.

## Matched primary-arm effects

Differences below are paired within the {manifest_hashes['matched_primary_cell_n']} matched cells and use a 20,000-replicate bootstrap clustered by 12 SAE features (seed 20260823).

{md_effect_table(effects)}

The strict result is a target-quality tradeoff. `dirfix` has substantially higher target success, but worse degeneration control. The intervals for coherence, prompt relevance, safety/integrity, and overall quality include zero. The small positive interval for the five-dimension mean combines opposing target and degeneration effects and must not be presented as across-the-board text-quality improvement.

## Paired analysis mechanically derived from blind item scores

No direct pair-annotation pass, second judgment, or pair-specific rationale was performed. The following are deterministic comparisons of the already-saved blind item scores for {manifest_hashes['matched_primary_cell_n']} matched cells: overall compares `overall_quality`; fluency compares the mean of coherence and degeneration control; concept compares `target_success`; exact equality is a tie. The rules were frozen before unblinding, after item scoring.

{md_preference_table(preferences)}

These are derived matched-cell comparisons, not 180 pair judgments. They favor `dirfix` for target concept, favor `naive` for the fluency composite, and do not give a conclusive overall preference because the clustered interval for the decisive overall win share includes 50%.

## Exploratory score associations

{md_association_table(associations)}

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
"""
    (root / "summary.md").write_text(summary, encoding="utf-8", newline="\n")

    readme = """# Task 1 AI annotation package

This directory contains an item-to-arm-blind annotation of 468 real Task 1 continuations by one AI subagent in a separate agent context, not by humans. The exact runtime backend model identifier was not exposed. Human inter-rater reliability is absent.

Read `summary.md` for results and limitations. `protocol.md` is the original locally frozen rubric; `operationalization.md` maps it to Task 1; `protocol_deviations.md` records the pre-unblinding amendment. The delivered paired analysis is mechanically derived from blind item scores for 180 matched cells; it is not a direct pair-judgment pass.

Machine-readable labels and joins are in `annotations_blind.jsonl`, `annotations.jsonl`, `annotations_with_masked_arm.*`, and `annotations_unblinded.*`. Aggregates and confidence intervals are in the summary CSV files. Sampling manifests, sealed mapping, provenance, hashes, and scripts are included so the deterministic parts can be checked. Interactive AI labels themselves are not claimed deterministic or independently reproducible.

Portable source locators are relative to the Task 1 repository. Machine-specific absolute paths, when retained, appear only in `provenance_machine_local.json`.
"""
    (root / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "summary": "summary.md",
                "readme": "README.md",
                "annotated_output_n": len(annotated),
                "matched_primary_cell_n": int(preferences[0]["matched_cell_n"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
