"""Build portable provenance records and package checksums."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def file_record(path: Path, locator: str, scope: str, role: str) -> dict:
    data = path.read_bytes()
    return {
        "path": locator,
        "path_scope": scope,
        "role": role,
        "bytes": len(data),
        "line_count": data.count(b"\n"),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", help="Task 1 repo root, or workspace containing 01-mech-interp")
    parser.add_argument("--annotation-dir", default=str(HERE))
    args = parser.parse_args()
    annotation_dir = Path(args.annotation_dir).resolve()
    repo = discover_repo_root(args.repo_root)
    workspace = repo.parent

    sources = [
        file_record(repo / "results" / "gen_r3.jsonl", "results/gen_r3.jsonl", "task_repo_relative", "current round-3 primary generated outputs"),
        file_record(repo / "results" / "gen_r3_ctrl.jsonl", "results/gen_r3_ctrl.jsonl", "task_repo_relative", "current round-3 random-rotation control outputs"),
        file_record(repo / "configs" / "features.yaml", "configs/features.yaml", "task_repo_relative", "test_r3 SAE feature target lists"),
        file_record(
            annotation_dir / "source_snapshot_task_README_before_annotation.md",
            "evaluation/ai_annotation/source_snapshot_task_README_before_annotation.md",
            "task_repo_relative",
            "exact snapshot of Task 1 repository overview read before annotation",
        ),
        file_record(
            annotation_dir / "source_snapshot_task_REPORT_before_annotation.md",
            "evaluation/ai_annotation/source_snapshot_task_REPORT_before_annotation.md",
            "task_repo_relative",
            "exact snapshot of Task 1 report read before annotation",
        ),
        file_record(
            annotation_dir / "source_snapshot_workspace_README_before_annotation.md",
            "evaluation/ai_annotation/source_snapshot_workspace_README_before_annotation.md",
            "task_repo_relative",
            "exact snapshot of workspace overview read before annotation",
        ),
        file_record(repo / "TASK.md", "TASK.md", "task_repo_relative", "official Task 1 statement"),
    ]
    with (annotation_dir / "source_hashes.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sources[0]))
        writer.writeheader()
        writer.writerows(sources)

    protocol_hash = sha256(annotation_dir / "protocol.md")
    prereg = json.loads((annotation_dir / "preregistration_record.json").read_text(encoding="utf-8"))
    if protocol_hash != prereg["protocol_sha256_at_preregistration"]:
        raise RuntimeError("Frozen protocol hash mismatch")

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "annotator_disclosure": {
            "identity": "one AI subagent in a separate agent context",
            "agent_context": "/root/task01_ai_annotator",
            "is_human": False,
            "human_annotator_count": 0,
            "human_inter_rater_reliability": None,
            "runtime_backend_model_identifier": "not exposed to the annotator",
        },
        "rubric_freeze": {
            "kind": "local timestamp-and-hash freeze; not an external registry",
            "protocol_sha256": protocol_hash,
            "timestamp_local": prereg["timestamp_local"],
            "before_method_mapping_inspection": True,
        },
        "blinding": {
            "item_to_arm_mapping_hidden_during_all_468_item judgments": True,
            "automatic_score_files_not_used_for_item_judgments": True,
            "hypothesis_and_result_blind": False,
            "limitation": "The current README/REPORT were read as instructed before candidate scoring, so the annotator knew the claimed method-level hypothesis/results but not which randomized item belonged to which arm.",
        },
        "sampling": {
            "frame": "all current non-stale rows in results/gen_r3.jsonl and results/gen_r3_ctrl.jsonl",
            "frame_n": 7560,
            "annotated_output_n": 468,
            "matched_primary_cell_n": 180,
            "features": 12,
            "seed": "task01-ai-annotation-20260823-v1",
            "manifest_sha256": sha256(annotation_dir / "blinded_manifest.jsonl"),
            "pair_manifest_sha256": sha256(annotation_dir / "blinded_pairs.jsonl"),
        },
        "source_files": sources,
        "independence_limits": [
            "The target keyword/top-token lists are SAE-derived and are not independent concept ground truth.",
            "Fluency and prompt relevance are more external linguistic checks, but still come from one AI judge.",
            "No human labels, human inter-rater reliability, or human-AI agreement are available.",
            "The deterministic manifest and aggregation are reproducible; interactive AI judgments are not claimed deterministic.",
        ],
        "paired_analysis": "180 matched cells mechanically derived from blind item scores; no direct pair-annotation pass and no pair-specific rationales",
    }
    (annotation_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    machine_local = {
        "warning": "Machine-local absolute paths are intentionally omitted from the public release copy",
        "workspace_absolute": None,
        "task_repo_absolute": None,
        "annotation_dir_absolute": None,
    }
    (annotation_dir / "provenance_machine_local.json").write_text(
        json.dumps(machine_local, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # SHA256SUMS intentionally excludes itself and mutable machine-local provenance.
    excluded = {"SHA256SUMS.txt", "provenance_machine_local.json"}
    files = sorted(
        p for p in annotation_dir.iterdir() if p.is_file() and p.name not in excluded
    )
    lines = [f"{sha256(path)}  {path.name}" for path in files]
    (annotation_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"source_file_n": len(sources), "checksummed_artifact_n": len(files), "protocol_sha256": protocol_hash}, indent=2))


if __name__ == "__main__":
    main()
