"""Create a deterministic, method-blinded annotation sample for Task 1.

This script writes only inside its own directory. It never mutates source artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
SEED = "task01-ai-annotation-20260823-v1"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_c(value: float) -> str:
    return format(float(value), ".12g")


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


def read_jsonl(path: Path, repo_root: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_file"] = path.relative_to(repo_root).as_posix()
            row["_source_line"] = line_no
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", help="Task 1 repo root, or workspace containing 01-mech-interp")
    args = parser.parse_args()
    task = discover_repo_root(args.repo_root)
    results = task / "results"
    primary_file = results / "gen_r3.jsonl"
    control_file = results / "gen_r3_ctrl.jsonl"
    features_file = task / "configs" / "features.yaml"

    feature_cfg = yaml.safe_load(features_file.read_text(encoding="utf-8"))["test_r3"]
    feat = {int(x["index"]): x for x in feature_cfg}
    feature_ids = sorted(feat)
    if len(feature_ids) != 12:
        raise RuntimeError(f"Expected 12 test_r3 features, found {len(feature_ids)}")

    rows = read_jsonl(primary_file, task) + read_jsonl(control_file, task)
    expected_frame = {"naive": 2880, "dirfix": 2880, "randrot": 1800}
    actual_frame = Counter(str(r["arm"]) for r in rows)
    if actual_frame != Counter(expected_frame):
        raise RuntimeError(f"Unexpected sampling frame: {actual_frame}")

    by_cell: dict[tuple[int, str, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (int(row["feature"]), canonical_c(row["c"]), int(row["prompt_idx"]))
        arm = str(row["arm"])
        if arm in by_cell[key]:
            raise RuntimeError(f"Duplicate arm in cell {key}: {arm}")
        by_cell[key][arm] = row

    # Choose the prompt indices without reference to method or generated text.
    primary_strengths = ["0", "0.5", "1", "1.25", "1.5", "2", "2.5", "3"]
    control_strengths = {"0", "1", "1.5", "2", "3"}
    selected_cells: list[tuple[int, str, int]] = []
    for feature in feature_ids:
        for c in primary_strengths:
            k = 1 if c == "0" else 2
            candidates = list(range(30))
            candidates.sort(key=lambda p: sha(f"{SEED}|select|{feature}|{c}|{p}"))
            for prompt_idx in candidates[:k]:
                selected_cells.append((feature, c, prompt_idx))

    # Stable opaque feature and pair codes, independent of methods and outputs.
    feature_order = sorted(feature_ids, key=lambda x: sha(f"{SEED}|feature|{x}"))
    feature_code = {feature: f"F{i:02d}" for i, feature in enumerate(feature_order, 1)}
    pair_order = sorted(selected_cells, key=lambda x: sha(f"{SEED}|pair|{x[0]}|{x[1]}|{x[2]}"))
    pair_code = {cell: f"P{i:04d}" for i, cell in enumerate(pair_order, 1)}

    # Alias assignment is deterministic but is stored only in the sealed key.
    arms = sorted(expected_frame, key=lambda a: sha(f"{SEED}|arm-alias|{a}"))
    arm_alias = {arm: f"MASK_{chr(65 + i)}" for i, arm in enumerate(arms)}

    selected: list[dict] = []
    for cell in selected_cells:
        feature, c, prompt_idx = cell
        available = by_cell[cell]
        for arm in ("naive", "dirfix"):
            if arm not in available:
                raise RuntimeError(f"Missing {arm} for {cell}")
            selected.append(available[arm])
        if c in control_strengths:
            if "randrot" not in available:
                raise RuntimeError(f"Missing randrot for {cell}")
            selected.append(available["randrot"])

    # Randomized item order and opaque item IDs.
    selected.sort(
        key=lambda r: sha(
            f"{SEED}|item|{Path(r['_source_file']).name}|{r['_source_line']}|"
            f"{r['feature']}|{canonical_c(r['c'])}|{r['prompt_idx']}|{r['arm']}"
        )
    )
    item_id_by_locator: dict[tuple[str, int], str] = {}
    blinded: list[dict] = []
    sealed: list[dict] = []
    for i, row in enumerate(selected, 1):
        item_id = f"A{i:04d}"
        source_file = str(row["_source_file"])
        source_line = int(row["_source_line"])
        item_id_by_locator[(source_file, source_line)] = item_id
        feature = int(row["feature"])
        c = canonical_c(row["c"])
        cell = (feature, c, int(row["prompt_idx"]))
        stable_locator = f"{source_file}:{source_line}"
        blinded.append(
            {
                "item_id": item_id,
                "pair_id": pair_code[cell],
                "feature_code": feature_code[feature],
                "strength": float(row["c"]),
                "prompt_idx": int(row["prompt_idx"]),
                "target_keywords": feat[feature].get("keywords", []),
                "target_top_tokens": feat[feature].get("top_tokens", []),
                "prompt": row["prompt"],
                "output": row["text"],
            }
        )
        sealed.append(
            {
                "item_id": item_id,
                "pair_id": pair_code[cell],
                "masked_arm": arm_alias[str(row["arm"])],
                "arm": row["arm"],
                "feature": feature,
                "feature_code": feature_code[feature],
                "strength": float(row["c"]),
                "prompt_idx": int(row["prompt_idx"]),
                "source_file": source_file,
                "source_line": source_line,
                "stable_locator_sha256": sha(stable_locator),
                "output_sha256": sha(str(row["text"])),
            }
        )

    expected_n = 468
    if len(blinded) != expected_n:
        raise RuntimeError(f"Expected N={expected_n}, got {len(blinded)}")

    # Primary blind pairs: the two non-control arms, with within-pair left/right order randomized.
    primary_pairs: list[dict] = []
    sealed_by_item = {r["item_id"]: r for r in sealed}
    blind_by_item = {r["item_id"]: r for r in blinded}
    for cell in pair_order:
        feature, c, prompt_idx = cell
        source_rows = by_cell[cell]
        ids = [
            item_id_by_locator[(str(source_rows[arm]["_source_file"]), int(source_rows[arm]["_source_line"]))]
            for arm in ("naive", "dirfix")
        ]
        ids.sort(key=lambda item: sha(f"{SEED}|pair-side|{pair_code[cell]}|{item}"))
        left, right = (blind_by_item[x] for x in ids)
        primary_pairs.append(
            {
                "pair_id": pair_code[cell],
                "feature_code": feature_code[feature],
                "strength": float(c),
                "prompt_idx": prompt_idx,
                "target_keywords": feat[feature].get("keywords", []),
                "prompt": left["prompt"],
                "left_item_id": left["item_id"],
                "left_output": left["output"],
                "right_item_id": right["item_id"],
                "right_output": right["output"],
            }
        )
        if {sealed_by_item[x]["arm"] for x in ids} != {"naive", "dirfix"}:
            raise RuntimeError(f"Bad primary pair {cell}")

    write_jsonl(HERE / "blinded_manifest.jsonl", blinded)
    write_jsonl(HERE / "sealed_mapping.jsonl", sealed)
    write_jsonl(HERE / "blinded_pairs.jsonl", primary_pairs)
    (HERE / "unblinding_key.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "arm_to_masked_alias": arm_alias,
                "feature_to_opaque_code": {str(k): v for k, v in feature_code.items()},
                "note": "Created before scoring; do not inspect during item or pair annotation.",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    frame_rows: list[dict] = []
    for path in (primary_file, control_file):
        relative_path = path.relative_to(task).as_posix()
        path_rows = [r for r in rows if r["_source_file"] == relative_path]
        grouped: Counter[tuple[str, str]] = Counter(
            (str(r["arm"]), canonical_c(r["c"])) for r in path_rows
        )
        for (arm, c), n in sorted(grouped.items()):
            frame_rows.append(
                {
                    "source_file": relative_path,
                    "arm": arm,
                    "strength": c,
                    "frame_n": n,
                    "sample_n": sum(
                        1
                        for r in sealed
                        if r["source_file"] == relative_path
                        and r["arm"] == arm
                        and canonical_c(r["strength"]) == c
                    ),
                }
            )
    with (HERE / "sampling_frame.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)

    manifest_hash = hashlib.sha256((HERE / "blinded_manifest.jsonl").read_bytes()).hexdigest()
    pair_hash = hashlib.sha256((HERE / "blinded_pairs.jsonl").read_bytes()).hexdigest()
    (HERE / "manifest_hashes.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "blinded_manifest_sha256": manifest_hash,
                "blinded_pairs_sha256": pair_hash,
                "sampling_frame_n": len(rows),
                "annotated_output_n": len(blinded),
                "matched_primary_cell_n": len(primary_pairs),
                "source_sha256": {
                    primary_file.relative_to(task).as_posix(): hashlib.sha256(primary_file.read_bytes()).hexdigest(),
                    control_file.relative_to(task).as_posix(): hashlib.sha256(control_file.read_bytes()).hexdigest(),
                    features_file.relative_to(task).as_posix(): hashlib.sha256(features_file.read_bytes()).hexdigest(),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "frame_n": len(rows),
                "sample_n": len(blinded),
                "matched_cell_n": len(primary_pairs),
                "manifest_sha256": manifest_hash,
                "pair_sha256": pair_hash,
                "masked_arm_counts": Counter(r["masked_arm"] for r in sealed),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
