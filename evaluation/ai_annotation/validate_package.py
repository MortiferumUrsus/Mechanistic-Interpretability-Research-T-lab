"""Read-only integrity validator for the Task 1 AI-annotation package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath

import yaml


HERE = Path(__file__).resolve().parent
SCORE_FIELDS = [
    "coherence_fluency",
    "prompt_relevance",
    "target_success",
    "degeneration_control",
    "safety_integrity",
    "overall_quality",
]
ALLOWED_FLAGS = {
    "blank_or_nontext",
    "truncated_or_incomplete",
    "language_mismatch",
    "prompt_or_control_leak",
    "obvious_encoding_or_markup_corruption",
    "severe_repetition",
    "source_contradiction",
    "unsupported_specific_claim",
    "material_safety_concern",
    "near_copy_when_change_required",
    "other_defect",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def discover_repo_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).resolve()
        if (root / "01-mech-interp" / "results" / "gen_r3.jsonl").is_file():
            root = root / "01-mech-interp"
        return root
    for candidate in (HERE, *HERE.parents):
        if (candidate / "results" / "gen_r3.jsonl").is_file():
            return candidate
        nested = candidate / "01-mech-interp"
        if (nested / "results" / "gen_r3.jsonl").is_file():
            return nested
    raise RuntimeError("Cannot discover Task 1 repo root; pass --repo-root")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_relative_locator(value: str, label: str) -> None:
    posix = PurePosixPath(value)
    require(not posix.is_absolute(), f"Absolute {label}: {value}")
    require(".." not in posix.parts, f"Parent traversal in {label}: {value}")
    require(not re.match(r"^[A-Za-z]:", value), f"Drive-qualified {label}: {value}")
    require("\\" not in value, f"Non-portable backslash in {label}: {value}")


def validate_checksums(root: Path) -> int:
    checksum_file = root / "SHA256SUMS.txt"
    require(checksum_file.is_file(), "Missing SHA256SUMS.txt")
    records: dict[str, str] = {}
    for line_no, line in enumerate(checksum_file.read_text(encoding="utf-8").splitlines(), 1):
        require("  " in line, f"Malformed SHA256SUMS line {line_no}")
        digest, name = line.split("  ", 1)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"Bad digest line {line_no}")
        require(Path(name).name == name, f"Checksum entry is not a basename: {name}")
        require(name not in records, f"Duplicate checksum entry: {name}")
        records[name] = digest
    expected_names = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name not in {"SHA256SUMS.txt", "provenance_machine_local.json"}
    }
    require(set(records) == expected_names, "SHA256SUMS coverage differs from portable package files")
    for name, expected in records.items():
        require(sha256(root / name) == expected, f"Package checksum mismatch: {name}")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", help="Task 1 repo root, or workspace containing 01-mech-interp")
    parser.add_argument("--annotation-dir", default=str(HERE))
    args = parser.parse_args()
    root = Path(args.annotation_dir).resolve()
    repo = discover_repo_root(args.repo_root)
    workspace = repo.parent
    warnings: list[str] = []

    required_files = {
        "README.md",
        "summary.md",
        "summary.json",
        "protocol.md",
        "operationalization.md",
        "protocol_deviations.md",
        "preregistration_record.json",
        "blinded_manifest.jsonl",
        "blinded_pairs.jsonl",
        "sealed_mapping.jsonl",
        "unblinding_key.json",
        "manifest_hashes.json",
        "sampling_frame.csv",
        "annotations_blind.jsonl",
        "annotations.jsonl",
        "annotations_with_masked_arm.jsonl",
        "annotations_unblinded.jsonl",
        "derived_pair_preferences_blind.jsonl",
        "derived_pair_preferences_unblinded.csv",
        "derived_pair_preference_summary_unblinded.csv",
        "method_summary_unblinded.csv",
        "paired_effects_unblinded.csv",
        "associations.csv",
        "source_hashes.csv",
        "provenance.json",
        "provenance_machine_local.json",
        "prepare_blind_manifest.py",
        "aggregate_annotations.py",
        "build_public_summary.py",
        "build_provenance.py",
        "validate_package.py",
        "SHA256SUMS.txt",
    }
    missing = sorted(name for name in required_files if not (root / name).is_file())
    require(not missing, f"Missing required package files: {missing}")

    prereg = json.loads((root / "preregistration_record.json").read_text(encoding="utf-8"))
    protocol_hash = sha256(root / "protocol.md")
    require(
        protocol_hash == prereg["protocol_sha256_at_preregistration"],
        "protocol.md differs from its locally frozen hash",
    )
    require(prereg.get("external_registry") is False, "Freeze record must deny external registry")

    hashes = json.loads((root / "manifest_hashes.json").read_text(encoding="utf-8"))
    require(hashes["seed"] == "task01-ai-annotation-20260823-v1", "Unexpected manifest seed")
    require(hashes["sampling_frame_n"] == 7560, "Unexpected sampling frame N")
    require(hashes["annotated_output_n"] == 468, "Unexpected annotated output N")
    require(hashes["matched_primary_cell_n"] == 180, "Unexpected matched-cell N")
    require(
        sha256(root / "blinded_manifest.jsonl") == hashes["blinded_manifest_sha256"],
        "Blind item manifest hash mismatch",
    )
    require(
        sha256(root / "blinded_pairs.jsonl") == hashes["blinded_pairs_sha256"],
        "Blind left/right manifest hash mismatch",
    )
    expected_critical_sources = {
        "results/gen_r3.jsonl",
        "results/gen_r3_ctrl.jsonl",
        "configs/features.yaml",
    }
    require(set(hashes["source_sha256"]) == expected_critical_sources, "Critical source hash set differs")
    for locator, digest in hashes["source_sha256"].items():
        validate_relative_locator(locator, "critical source locator")
        require(sha256(repo / PurePosixPath(locator)) == digest, f"Critical source drift: {locator}")

    manifest = read_jsonl(root / "blinded_manifest.jsonl")
    annotations = read_jsonl(root / "annotations_blind.jsonl")
    mapping = read_jsonl(root / "sealed_mapping.jsonl")
    pairs = read_jsonl(root / "blinded_pairs.jsonl")
    require(len(manifest) == 468, "Blind manifest is not N=468")
    require(len(annotations) == 468, "Blind annotations are not N=468")
    require(len(mapping) == 468, "Sealed mapping is not N=468")
    require(len(pairs) == 180, "Matched-cell display manifest is not N=180")
    manifest_by_id = {row["item_id"]: row for row in manifest}
    score_by_id = {row["item_id"]: row for row in annotations}
    map_by_id = {row["item_id"]: row for row in mapping}
    require(len(manifest_by_id) == 468, "Duplicate manifest item ID")
    require(len(score_by_id) == 468, "Duplicate annotation item ID")
    require(len(map_by_id) == 468, "Duplicate mapping item ID")
    require(set(manifest_by_id) == set(score_by_id) == set(map_by_id), "Item ID sets differ")
    require(len({row["pair_id"] for row in pairs}) == 180, "Duplicate matched-cell ID")

    for row in annotations:
        for field in SCORE_FIELDS:
            require(type(row.get(field)) is int and 1 <= row[field] <= 5, f"Bad {field}: {row['item_id']}")
        require(isinstance(row.get("defect_flags"), list), f"Flags not a list: {row['item_id']}")
        require(not (set(row["defect_flags"]) - ALLOWED_FLAGS), f"Unknown flag: {row['item_id']}")
        require(len(row["defect_flags"]) == len(set(row["defect_flags"])), f"Duplicate flag: {row['item_id']}")
        require(isinstance(row.get("rationale"), str) and row["rationale"].strip(), f"Missing rationale: {row['item_id']}")

    feature_cfg = yaml.safe_load((repo / "configs" / "features.yaml").read_text(encoding="utf-8"))[
        "test_r3"
    ]
    feature_by_id = {int(row["index"]): row for row in feature_cfg}
    source_cache: dict[str, list[str]] = {}
    for locator in ("results/gen_r3.jsonl", "results/gen_r3_ctrl.jsonl"):
        source_cache[locator] = (repo / PurePosixPath(locator)).read_text(encoding="utf-8").splitlines()

    for item_id, item in manifest_by_id.items():
        mapped = map_by_id[item_id]
        checks = [
            item["pair_id"] == mapped["pair_id"],
            item["feature_code"] == mapped["feature_code"],
            float(item["strength"]) == float(mapped["strength"]),
            int(item["prompt_idx"]) == int(mapped["prompt_idx"]),
            text_sha256(item["output"]) == mapped["output_sha256"],
        ]
        require(all(checks), f"Manifest/mapping disagreement: {item_id}")
        locator = mapped["source_file"]
        validate_relative_locator(locator, "mapping source locator")
        require(locator in source_cache, f"Unexpected mapping source: {locator}")
        line_no = int(mapped["source_line"])
        require(1 <= line_no <= len(source_cache[locator]), f"Bad source line: {item_id}")
        source = json.loads(source_cache[locator][line_no - 1])
        require(source["text"] == item["output"], f"Source output differs: {item_id}")
        require(source["prompt"] == item["prompt"], f"Source prompt differs: {item_id}")
        require(source["arm"] == mapped["arm"], f"Source arm differs: {item_id}")
        require(int(source["feature"]) == int(mapped["feature"]), f"Source feature differs: {item_id}")
        require(float(source["c"]) == float(mapped["strength"]), f"Source strength differs: {item_id}")
        require(int(source["prompt_idx"]) == int(mapped["prompt_idx"]), f"Source prompt index differs: {item_id}")
        stable = f"{locator}:{line_no}"
        require(text_sha256(stable) == mapped["stable_locator_sha256"], f"Locator hash differs: {item_id}")
        feature = feature_by_id[int(mapped["feature"])]
        require(item["target_keywords"] == feature.get("keywords", []), f"Keyword target differs: {item_id}")
        require(item["target_top_tokens"] == feature.get("top_tokens", []), f"Token target differs: {item_id}")

    arm_counts = Counter(row["arm"] for row in mapping)
    require(arm_counts == Counter({"dirfix": 180, "naive": 180, "randrot": 108}), f"Arm counts differ: {arm_counts}")
    key = json.loads((root / "unblinding_key.json").read_text(encoding="utf-8"))
    alias = key["arm_to_masked_alias"]
    require(alias == {"dirfix": "MASK_A", "randrot": "MASK_B", "naive": "MASK_C"}, "Unexpected masked-arm key")
    for row in mapping:
        require(row["masked_arm"] == alias[row["arm"]], f"Masked arm differs: {row['item_id']}")

    pair_items: list[str] = []
    for pair in pairs:
        left_id, right_id = pair["left_item_id"], pair["right_item_id"]
        require(left_id != right_id, f"Duplicate sides: {pair['pair_id']}")
        require(left_id in manifest_by_id and right_id in manifest_by_id, f"Unknown side: {pair['pair_id']}")
        pair_items.extend([left_id, right_id])
        for side, item_id in (("left", left_id), ("right", right_id)):
            item, mapped = manifest_by_id[item_id], map_by_id[item_id]
            require(item["pair_id"] == pair["pair_id"] == mapped["pair_id"], f"Pair ID mismatch: {pair['pair_id']}")
            require(item["feature_code"] == pair["feature_code"] == mapped["feature_code"], f"Pair feature mismatch: {pair['pair_id']}")
            require(float(item["strength"]) == float(pair["strength"]) == float(mapped["strength"]), f"Pair strength mismatch: {pair['pair_id']}")
            require(int(item["prompt_idx"]) == int(pair["prompt_idx"]) == int(mapped["prompt_idx"]), f"Pair prompt index mismatch: {pair['pair_id']}")
            require(item["prompt"] == pair["prompt"], f"Pair prompt text mismatch: {pair['pair_id']}")
            require(item["output"] == pair[f"{side}_output"], f"Pair output mismatch: {pair['pair_id']}")
        require({map_by_id[left_id]["arm"], map_by_id[right_id]["arm"]} == {"naive", "dirfix"}, f"Bad primary arms: {pair['pair_id']}")
    require(len(pair_items) == len(set(pair_items)) == 360, "Primary item is missing or repeated across matched cells")

    frame = read_csv(root / "sampling_frame.csv")
    require(sum(int(row["frame_n"]) for row in frame) == 7560, "Sampling frame CSV total differs")
    require(sum(int(row["sample_n"]) for row in frame) == 468, "Sampling sample total differs")
    for row in frame:
        validate_relative_locator(row["source_file"], "sampling frame source locator")

    derived = read_jsonl(root / "derived_pair_preferences_blind.jsonl")
    require(len(derived) == 180 and len({row["pair_id"] for row in derived}) == 180, "Derived matched-cell rows differ")
    for row in derived:
        left, right = score_by_id[row["left_item_id"]], score_by_id[row["right_item_id"]]
        expected = {
            "overall_preference": "left" if left["overall_quality"] > right["overall_quality"] else "right" if right["overall_quality"] > left["overall_quality"] else "tie",
            "fluency_preference": "left" if left["coherence_fluency"] + left["degeneration_control"] > right["coherence_fluency"] + right["degeneration_control"] else "right" if right["coherence_fluency"] + right["degeneration_control"] > left["coherence_fluency"] + left["degeneration_control"] else "tie",
            "concept_preference": "left" if left["target_success"] > right["target_success"] else "right" if right["target_success"] > left["target_success"] else "tie",
        }
        for field, value in expected.items():
            require(row[field] == value, f"Derived preference differs: {row['pair_id']} {field}")
        require(row["derivation"] == "mechanical_from_blind_item_scores", f"Bad derivation label: {row['pair_id']}")

    method_summary = read_csv(root / "method_summary_unblinded.csv")
    require({row["group"]: int(row["n"]) for row in method_summary} == {"ALL": 468, "dirfix": 180, "naive": 180, "randrot": 108}, "Method summary N differs")
    effects = read_csv(root / "paired_effects_unblinded.csv")
    require(len(effects) == 7, "Expected seven paired-effect outcomes")
    for row in effects:
        require(int(row["matched_cell_n"]) == 180 and int(row["feature_cluster_n"]) == 12, f"Bad effect design: {row['outcome']}")
        require(int(row["bootstrap_seed"]) == 20260823 and int(row["bootstrap_reps"]) == 20000, f"Bad effect bootstrap: {row['outcome']}")
    pref_summary = read_csv(root / "derived_pair_preference_summary_unblinded.csv")
    require({row["dimension"] for row in pref_summary} == {"overall", "fluency", "concept"}, "Preference dimensions differ")
    for row in pref_summary:
        require(int(row["dirfix_wins"]) + int(row["naive_wins"]) + int(row["ties"]) == 180, f"Preference count differs: {row['dimension']}")
        require("not direct pair judgments" in row["note"], f"Missing derived-only disclosure: {row['dimension']}")
    associations = read_csv(root / "associations.csv")
    require(len(associations) == 12, "Expected 12 exploratory association rows")

    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    disclosure = provenance["annotator_disclosure"]
    require(disclosure["identity"] == "one AI subagent in a separate agent context", "Bad AI disclosure")
    require(disclosure["is_human"] is False and disclosure["human_annotator_count"] == 0, "Human disclosure differs")
    require(disclosure["human_inter_rater_reliability"] is None, "Human IRR must be absent")
    require(disclosure["runtime_backend_model_identifier"] == "not exposed to the annotator", "Backend model disclosure differs")
    require(provenance["sampling"]["matched_primary_cell_n"] == 180, "Provenance matched-cell N differs")
    require(provenance["sampling"]["manifest_sha256"] == hashes["blinded_manifest_sha256"], "Provenance manifest hash differs")

    source_records = read_csv(root / "source_hashes.csv")
    provenance_sources = {row["path_scope"] + ":" + row["path"]: row for row in provenance["source_files"]}
    csv_sources = {row["path_scope"] + ":" + row["path"]: row for row in source_records}
    require(set(provenance_sources) == set(csv_sources), "Source provenance record sets differ")
    for key_name, record in provenance_sources.items():
        csv_record = csv_sources[key_name]
        for field in ("path", "path_scope", "role", "sha256"):
            require(str(record[field]) == str(csv_record[field]), f"Source provenance differs: {key_name} {field}")
        validate_relative_locator(record["path"], "provenance source locator")
        base = repo if record["path_scope"] == "task_repo_relative" else workspace
        current = base / PurePosixPath(record["path"])
        require(current.is_file(), f"Provenance source missing: {key_name}")
        if record["role"] in {
            "current round-3 primary generated outputs",
            "current round-3 random-rotation control outputs",
            "test_r3 SAE feature target lists",
        }:
            require(sha256(current) == record["sha256"], f"Critical provenance source drift: {key_name}")
        elif sha256(current) != record["sha256"]:
            warnings.append(f"historical narrative/document source changed after annotation: {key_name}")

    # Public artifacts must not leak this machine's absolute workspace path.
    machine_root = str(workspace)
    for path in root.iterdir():
        if not path.is_file() or path.name == "provenance_machine_local.json":
            continue
        if path.suffix.lower() not in {".py", ".md", ".json", ".jsonl", ".csv", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        require(machine_root not in text, f"Machine-local absolute path leaked into portable artifact: {path.name}")

    checksum_n = validate_checksums(root)
    result = {
        "status": "PASS",
        "validated_annotation_n": len(annotations),
        "validated_matched_cell_n": len(pairs),
        "sampling_frame_n": hashes["sampling_frame_n"],
        "feature_n": len(feature_by_id),
        "protocol_sha256": protocol_hash,
        "blinded_manifest_sha256": hashes["blinded_manifest_sha256"],
        "blinded_pairs_sha256": hashes["blinded_pairs_sha256"],
        "checksummed_artifact_n": checksum_n,
        "warnings": warnings,
    }
    print("VALIDATION PASS")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
