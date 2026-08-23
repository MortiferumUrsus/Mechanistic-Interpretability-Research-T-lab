"""Validate, unblind, aggregate, and export Task 1 AI annotations.

The AI labels are inputs. This script deterministically reproduces joins, derived
fields, paired preferences, summaries, bootstrap intervals, and associations.
It does not reproduce the interactive AI judgment process itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


BOOTSTRAP_SEED = 20260823
BOOTSTRAP_REPS = 20_000
ASSOCIATION_BOOTSTRAP_REPS = 5_000
SCORE_FIELDS = [
    "coherence_fluency",
    "prompt_relevance",
    "target_success",
    "degeneration_control",
    "safety_integrity",
    "overall_quality",
]
DIMENSION_FIELDS = SCORE_FIELDS[:5]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cooked = {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                for k, v in row.items()
            }
            writer.writerow(cooked)


def preference(left: float, right: float) -> str:
    if left > right:
        return "left"
    if right > left:
        return "right"
    return "tie"


def percentile_ci(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi)


def cluster_bootstrap_mean(
    values_by_feature: dict[int, list[float]], rng: np.random.Generator, reps: int
) -> tuple[float, float]:
    features = sorted(values_by_feature)
    estimates: list[float] = []
    for _ in range(reps):
        sampled = rng.choice(features, size=len(features), replace=True)
        values = [v for feature in sampled for v in values_by_feature[int(feature)]]
        estimates.append(float(np.mean(values)))
    return percentile_ci(estimates)


def cluster_bootstrap_ratio(
    decisive_by_feature: dict[int, list[int]], rng: np.random.Generator, reps: int
) -> tuple[float, float]:
    features = sorted(decisive_by_feature)
    estimates: list[float] = []
    for _ in range(reps):
        sampled = rng.choice(features, size=len(features), replace=True)
        values = [v for feature in sampled for v in decisive_by_feature[int(feature)]]
        if values:
            estimates.append(float(np.mean(values)))
    return percentile_ci(estimates)


def cluster_bootstrap_spearman(
    rows: list[dict], x: str, y: str, rng: np.random.Generator, reps: int
) -> tuple[float, float]:
    by_feature: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_feature[int(row["feature"])].append(row)
    features = sorted(by_feature)
    estimates: list[float] = []
    for _ in range(reps):
        sampled = rng.choice(features, size=len(features), replace=True)
        sample_rows = [row for feature in sampled for row in by_feature[int(feature)]]
        rho = float(spearmanr([r[x] for r in sample_rows], [r[y] for r in sample_rows]).statistic)
        if np.isfinite(rho):
            estimates.append(rho)
    return percentile_ci(estimates)


def summary_row(group: str, rows: list[dict]) -> dict:
    n = len(rows)
    result: dict = {"group": group, "n": n}
    for field in SCORE_FIELDS + ["dimension_mean"]:
        values = np.asarray([float(r[field]) for r in rows])
        result[f"{field}_mean"] = round(float(values.mean()), 6)
        result[f"{field}_median"] = round(float(np.median(values)), 6)
    thresholds = {
        "coherent": lambda r: r["coherence_fluency"] >= 4,
        "relevant": lambda r: r["prompt_relevance"] >= 4,
        "concept_present": lambda r: r["target_success"] >= 4,
        "concept_partial_or_ambiguous": lambda r: r["target_success"] == 3,
        "degeneration_or_repetition_issue": lambda r: r["degeneration_control"] <= 3,
        "severe_degeneration": lambda r: r["degeneration_control"] <= 2,
        "safety_or_integrity_issue": lambda r: r["safety_integrity"] <= 3,
    }
    for name, fn in thresholds.items():
        count = sum(bool(fn(r)) for r in rows)
        result[f"{name}_count"] = count
        result[f"{name}_share"] = round(count / n, 6)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    root = Path(args.annotation_dir).resolve()

    manifest = read_jsonl(root / "blinded_manifest.jsonl")
    annotations = read_jsonl(root / "annotations_blind.jsonl")
    mapping = read_jsonl(root / "sealed_mapping.jsonl")
    pairs = read_jsonl(root / "blinded_pairs.jsonl")

    manifest_ids = [r["item_id"] for r in manifest]
    annotation_ids = [r["item_id"] for r in annotations]
    mapping_ids = [r["item_id"] for r in mapping]
    if len(manifest) != 468 or len(set(manifest_ids)) != 468:
        raise RuntimeError("Blinded manifest must contain exactly 468 unique items")
    if len(annotations) != 468 or len(set(annotation_ids)) != 468:
        raise RuntimeError("Annotations must contain exactly 468 unique items")
    if set(annotation_ids) != set(manifest_ids) or set(mapping_ids) != set(manifest_ids):
        raise RuntimeError("Manifest/annotation/mapping ID sets differ")
    if len(pairs) != 180 or len({p["pair_id"] for p in pairs}) != 180:
        raise RuntimeError("Pair manifest must contain exactly 180 unique cells")
    for row in annotations:
        for field in SCORE_FIELDS:
            if row[field] not in (1, 2, 3, 4, 5):
                raise RuntimeError(f"Invalid score {row['item_id']} {field}={row[field]}")

    manifest_by_id = {r["item_id"]: r for r in manifest}
    score_by_id = {r["item_id"]: r for r in annotations}
    map_by_id = {r["item_id"]: r for r in mapping}
    for item_id in manifest_ids:
        source, mapped = manifest_by_id[item_id], map_by_id[item_id]
        if hashlib.sha256(source["output"].encode("utf-8")).hexdigest() != mapped["output_sha256"]:
            raise RuntimeError(f"Output hash mismatch for {item_id}")
        checks = (
            source["pair_id"] == mapped["pair_id"],
            source["feature_code"] == mapped["feature_code"],
            float(source["strength"]) == float(mapped["strength"]),
            int(source["prompt_idx"]) == int(mapped["prompt_idx"]),
        )
        if not all(checks):
            raise RuntimeError(f"Manifest/mapping metadata mismatch for {item_id}")
    prereg = json.loads((root / "preregistration_record.json").read_text(encoding="utf-8"))
    protocol_hash = hashlib.sha256((root / "protocol.md").read_bytes()).hexdigest()
    if protocol_hash != prereg["protocol_sha256_at_preregistration"]:
        raise RuntimeError("protocol.md changed after its frozen hash was recorded")

    joined: list[dict] = []
    blinded_export: list[dict] = []
    for item_id in manifest_ids:
        source = manifest_by_id[item_id]
        score = score_by_id[item_id]
        mapped = map_by_id[item_id]
        flags = set(score["defect_flags"])
        derived = {
            "concept_presence": (
                "present"
                if score["target_success"] >= 4
                else "partial_or_ambiguous"
                if score["target_success"] == 3
                else "absent"
            ),
            "dimension_mean": round(
                float(np.mean([score[field] for field in DIMENSION_FIELDS])), 6
            ),
            "coherent": score["coherence_fluency"] >= 4,
            "relevant": score["prompt_relevance"] >= 4,
            "concept_present": score["target_success"] >= 4,
            "degeneration_or_repetition_issue": score["degeneration_control"] <= 3,
            "severe_degeneration": score["degeneration_control"] <= 2,
            "safety_or_integrity_issue": score["safety_integrity"] <= 3,
            "truncated_or_incomplete": "truncated_or_incomplete" in flags,
        }
        blind_row = {**source, **{k: score[k] for k in SCORE_FIELDS}, **derived}
        blind_row["defect_flags"] = score["defect_flags"]
        blind_row["rationale"] = score["rationale"]
        blinded_export.append(blind_row)
        joined.append(
            {
                **blind_row,
                "masked_arm": mapped["masked_arm"],
                "arm": mapped["arm"],
                "feature": int(mapped["feature"]),
                "source_file": mapped["source_file"],
                "source_line": int(mapped["source_line"]),
                "output_sha256": mapped["output_sha256"],
            }
        )

    write_jsonl(root / "annotations.jsonl", blinded_export)
    write_csv(root / "annotations.csv", blinded_export)
    (root / "annotations.json").write_text(
        json.dumps(blinded_export, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    masked_export = [{k: v for k, v in row.items() if k != "arm"} for row in joined]
    write_csv(root / "annotations_with_masked_arm.csv", masked_export)
    write_jsonl(root / "annotations_with_masked_arm.jsonl", masked_export)
    write_csv(root / "annotations_unblinded.csv", joined)
    write_jsonl(root / "annotations_unblinded.jsonl", joined)

    # Derived blind pair preferences; not a second annotation pass.
    derived_pairs_blind: list[dict] = []
    derived_pairs_unblinded: list[dict] = []
    for pair in pairs:
        left_id, right_id = pair["left_item_id"], pair["right_item_id"]
        if left_id == right_id or left_id not in manifest_by_id or right_id not in manifest_by_id:
            raise RuntimeError(f"Invalid pair members in {pair['pair_id']}")
        for item_id in (left_id, right_id):
            item = manifest_by_id[item_id]
            mapped = map_by_id[item_id]
            checks = (
                item["pair_id"] == pair["pair_id"],
                item["feature_code"] == pair["feature_code"],
                float(item["strength"]) == float(pair["strength"]),
                int(item["prompt_idx"]) == int(pair["prompt_idx"]),
                mapped["pair_id"] == pair["pair_id"],
            )
            if not all(checks):
                raise RuntimeError(f"Pair/manifest metadata mismatch for {pair['pair_id']} {item_id}")
        if {map_by_id[left_id]["arm"], map_by_id[right_id]["arm"]} != {"naive", "dirfix"}:
            raise RuntimeError(f"Pair {pair['pair_id']} is not naive/dirfix")
        left, right = score_by_id[left_id], score_by_id[right_id]
        left_fluency = (left["coherence_fluency"] + left["degeneration_control"]) / 2
        right_fluency = (right["coherence_fluency"] + right["degeneration_control"]) / 2
        blind = {
            "pair_id": pair["pair_id"],
            "feature_code": pair["feature_code"],
            "strength": pair["strength"],
            "prompt_idx": pair["prompt_idx"],
            "left_item_id": left_id,
            "right_item_id": right_id,
            "left_overall_quality": left["overall_quality"],
            "right_overall_quality": right["overall_quality"],
            "left_fluency_composite": left_fluency,
            "right_fluency_composite": right_fluency,
            "left_target_success": left["target_success"],
            "right_target_success": right["target_success"],
            "overall_preference": preference(left["overall_quality"], right["overall_quality"]),
            "fluency_preference": preference(left_fluency, right_fluency),
            "concept_preference": preference(left["target_success"], right["target_success"]),
            "derivation": "mechanical_from_blind_item_scores",
        }
        derived_pairs_blind.append(blind)
        left_map, right_map = map_by_id[left_id], map_by_id[right_id]
        derived_pairs_unblinded.append(
            {
                **blind,
                "feature": int(left_map["feature"]),
                "left_masked_arm": left_map["masked_arm"],
                "right_masked_arm": right_map["masked_arm"],
                "left_arm": left_map["arm"],
                "right_arm": right_map["arm"],
            }
        )
    write_csv(root / "derived_pair_preferences_blind.csv", derived_pairs_blind)
    write_jsonl(root / "derived_pair_preferences_blind.jsonl", derived_pairs_blind)
    write_csv(root / "derived_pair_preferences_unblinded.csv", derived_pairs_unblinded)

    method_summary = [summary_row("ALL", joined)]
    for masked_arm in sorted({r["masked_arm"] for r in joined}):
        rows = [r for r in joined if r["masked_arm"] == masked_arm]
        method_summary.append(summary_row(masked_arm, rows))
    write_csv(root / "method_summary_masked.csv", method_summary)

    unblinded_summary = [summary_row("ALL", joined)]
    for arm in sorted({r["arm"] for r in joined}):
        rows = [r for r in joined if r["arm"] == arm]
        unblinded_summary.append(summary_row(arm, rows))
    write_csv(root / "method_summary_unblinded.csv", unblinded_summary)

    strength_summary: list[dict] = []
    for arm in sorted({r["arm"] for r in joined}):
        arm_rows = [r for r in joined if r["arm"] == arm]
        for strength in sorted({float(r["strength"]) for r in arm_rows}):
            rows = [r for r in arm_rows if float(r["strength"]) == strength]
            strength_summary.append({"arm": arm, "strength": strength, **summary_row("cell", rows)})
    write_csv(root / "strength_summary_unblinded.csv", strength_summary)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    primary_pairs = derived_pairs_unblinded
    if any({p["left_arm"], p["right_arm"]} != {"naive", "dirfix"} for p in primary_pairs):
        raise RuntimeError("Primary pairs do not contain exactly naive and dirfix")

    paired_effects: list[dict] = []
    for field in SCORE_FIELDS + ["dimension_mean"]:
        diffs_by_feature: dict[int, list[float]] = defaultdict(list)
        for pair in primary_pairs:
            values = {
                pair["left_arm"]: next(r[field] for r in joined if r["item_id"] == pair["left_item_id"]),
                pair["right_arm"]: next(r[field] for r in joined if r["item_id"] == pair["right_item_id"]),
            }
            diffs_by_feature[int(pair["feature"])].append(float(values["dirfix"] - values["naive"]))
        all_diffs = [v for values in diffs_by_feature.values() for v in values]
        lo, hi = cluster_bootstrap_mean(diffs_by_feature, rng, BOOTSTRAP_REPS)
        paired_effects.append(
            {
                "outcome": field,
                "direction": "dirfix_minus_naive",
                "matched_cell_n": len(all_diffs),
                "feature_cluster_n": len(diffs_by_feature),
                "mean_difference": round(float(np.mean(all_diffs)), 6),
                "median_difference": round(float(np.median(all_diffs)), 6),
                "cluster_bootstrap_ci95_low": round(lo, 6),
                "cluster_bootstrap_ci95_high": round(hi, 6),
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_reps": BOOTSTRAP_REPS,
            }
        )
    write_csv(root / "paired_effects_unblinded.csv", paired_effects)

    pref_summary: list[dict] = []
    for dimension in ("overall", "fluency", "concept"):
        pref_field = f"{dimension}_preference"
        winners: list[tuple[int, str]] = []
        for pair in primary_pairs:
            pref = pair[pref_field]
            if pref == "tie":
                winners.append((int(pair["feature"]), "tie"))
            elif pref == "left":
                winners.append((int(pair["feature"]), pair["left_arm"]))
            else:
                winners.append((int(pair["feature"]), pair["right_arm"]))
        counts = Counter(w for _, w in winners)
        decisive = [(feature, 1 if winner == "dirfix" else 0) for feature, winner in winners if winner != "tie"]
        decisive_by_feature: dict[int, list[int]] = defaultdict(list)
        for feature, value in decisive:
            decisive_by_feature[feature].append(value)
        share = float(np.mean([v for _, v in decisive])) if decisive else float("nan")
        lo, hi = cluster_bootstrap_ratio(decisive_by_feature, rng, BOOTSTRAP_REPS)
        pref_summary.append(
            {
                "dimension": dimension,
                "matched_cell_n": len(winners),
                "dirfix_wins": counts["dirfix"],
                "naive_wins": counts["naive"],
                "ties": counts["tie"],
                "decisive_n": len(decisive),
                "dirfix_win_share_among_decisive": round(share, 6),
                "cluster_bootstrap_ci95_low": round(lo, 6),
                "cluster_bootstrap_ci95_high": round(hi, 6),
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_reps": BOOTSTRAP_REPS,
                "note": "derived mechanically from blind item scores; not direct pair judgments",
            }
        )
    write_csv(root / "derived_pair_preference_summary_unblinded.csv", pref_summary)

    associations: list[dict] = []
    for group in ["ALL", *sorted({r["arm"] for r in joined})]:
        rows = joined if group == "ALL" else [r for r in joined if r["arm"] == group]
        for y in ("coherence_fluency", "prompt_relevance", "degeneration_control"):
            rho = float(spearmanr([r["target_success"] for r in rows], [r[y] for r in rows]).statistic)
            lo, hi = cluster_bootstrap_spearman(
                rows, "target_success", y, rng, ASSOCIATION_BOOTSTRAP_REPS
            )
            associations.append(
                {
                    "group": group,
                    "n": len(rows),
                    "x": "target_success",
                    "y": y,
                    "spearman_rho": round(rho, 6),
                    "feature_cluster_bootstrap_ci95_low": round(lo, 6),
                    "feature_cluster_bootstrap_ci95_high": round(hi, 6),
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_reps": ASSOCIATION_BOOTSTRAP_REPS,
                    "note": "exploratory; condition-balanced dependent sample",
                }
            )
    write_csv(root / "associations.csv", associations)

    arm_key = json.loads((root / "unblinding_key.json").read_text(encoding="utf-8"))[
        "arm_to_masked_alias"
    ]
    summary = {
        "disclosure": {
            "annotator": "one AI subagent in a separate agent context; not human",
            "human_inter_rater_reliability": None,
            "runtime_backend_model_identifier": "not exposed to the annotator",
            "target_ground_truth_limit": "target lists are SAE-derived and are not independent concept ground truth",
        },
        "sampling": {
            "frame_n": 7560,
            "annotated_output_n": 468,
            "matched_primary_cell_n": 180,
            "feature_n": 12,
            "manifest_seed": "task01-ai-annotation-20260823-v1",
            "blinded_manifest_sha256": hashlib.sha256(
                (root / "blinded_manifest.jsonl").read_bytes()
            ).hexdigest(),
        },
        "unblinding_key": arm_key,
        "method_summary": unblinded_summary,
        "paired_effects": paired_effects,
        "derived_pair_preference_summary": pref_summary,
        "associations": associations,
        "reproducibility_scope": (
            "The deterministic sample, joins, derived fields, aggregation, and intervals are reproducible. "
            "The interactive single-AI labeling judgments are not claimed to be deterministic or independently reproducible."
        ),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "validated_annotation_n": len(annotations),
                "validated_matched_cell_n": len(pairs),
                "arms": Counter(r["arm"] for r in joined),
                "masked_arms": Counter(r["masked_arm"] for r in joined),
                "arm_key": arm_key,
                "paired_effects": paired_effects,
                "pair_preferences": pref_summary,
            },
            ensure_ascii=False,
            indent=2,
            default=lambda x: dict(x),
        )
    )


if __name__ == "__main__":
    main()
