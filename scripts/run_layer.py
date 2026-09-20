"""Replicate the whole steering-direction-correction chain on a layer other than 7.

Round two found a shared low-rank correction `w(v) = normalize(v + M v)` on the SAE trained at
blocks.7.hook_resid_pre / blocks.6.hook_resid_post. This driver asks whether the same kind of
shared correction exists on another layer, and whether its shared direction `d_bar` agrees with
the one found at layer 7. `src/common.py` reads `TLAB_LAYER=L` to move SAE_ID / HOOK (and, through
them, train_ctr.LAYER and activations.py's identity check) to that layer; this script is the only
thing that sets it.

Meant to run in a clean environment (Kaggle: a fresh checkout, empty data/), but also runnable
locally against an isolated copy of the repository. It refuses to run the stages that overwrite
data/configs (select, dump, train_dir) against what looks like the live, already-populated
repository, unless told otherwise with --i-know-this-overwrites.

    python run_layer.py --layer 10
    python run_layer.py --layer 10 --stages prep,verify
    python run_layer.py --layer 10 --stages dump --n-tokens 20000 --root <isolated copy>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

STAGE_ORDER = (
    "prep",
    "verify",
    "dump",
    "prompts",
    "select",
    "train_dir",
    "anatomy",
    "gen",
    "note",
)
# These stages overwrite configs/features.yaml, data/splits.npz, data/acts.f16 or train a fresh
# checkpoint from them -- fine on a throwaway Kaggle checkout, destructive against the live repo.
GUARDED_STAGES = {"select", "dump", "train_dir"}
# Presence of this file means the live, round-2 repository (dirfix_vs_naive.csv summarises round
# two's dirfix-vs-naive comparison and is only ever written by a real run against the frozen
# layer-7 pipeline) -- not an isolated copy meant to be overwritten.
LIVE_REPO_MARKER = "dirfix_vs_naive.csv"


class Ctx:
    def __init__(self, layer: int, root: Path, python: str, n_tokens: int):
        self.layer = layer
        self.root = root
        self.src = root / "src"
        self.data = root / "data"
        self.results = root / "results"
        self.checkpoints = root / "checkpoints"
        self.configs = root / "configs"
        self.python = python
        self.n_tokens = n_tokens


def run_cmd(ctx: Ctx, args: list[str]) -> None:
    cmd = [ctx.python, *args]
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(ctx.src), check=True)


def _df_to_md(df) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for row in df.itertuples(index=False):
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


# --- stages -----------------------------------------------------------------------------------


def stage_prep(ctx: Ctx) -> None:
    ctx.data.mkdir(parents=True, exist_ok=True)
    ctx.results.mkdir(parents=True, exist_ok=True)

    # Snapshot the layer-7 shared direction under its own name before anything can touch it, so
    # later stages (and the "anatomy" stage in particular, which overwrites checkpoints/shared_direction.pt
    # with THIS layer's d_bar) still have the layer-7 reference to compare against.
    src_shared = ctx.checkpoints / "shared_direction.pt"
    dst_shared = ctx.checkpoints / "shared_direction_L7.pt"
    if src_shared.exists():
        shutil.copy2(src_shared, dst_shared)
        print(f"copied {src_shared} -> {dst_shared} (layer-7 reference)")
    else:
        print(f"WARNING: {src_shared} not found; no layer-7 reference to snapshot. Continuing.")

    # These hold latent indices selected on the OLD layer's SAE; on the new layer the same integer
    # indices point at unrelated latents, so train_direction.py must not silently exclude them as
    # if they were this layer's round-3/round-4 holdouts.
    for name in ("splits_r3.npz", "splits_r4.npz"):
        p = ctx.data / name
        if p.exists():
            p.unlink()
            print(f"removed {p} (round-3/4 holdout indices belong to the old layer)")
        else:
            print(f"{p} does not exist; nothing to remove")

    # Import only now: this module-level import reads TLAB_LAYER at import time, and main() has
    # already set it (and pointed sys.path at <root>/src) before calling any stage.
    import common

    manifest = (
        f"layer={ctx.layer}\n"
        f"HOOK={common.HOOK}\n"
        f"SAE_ID={common.SAE_ID}\n"
        f"root={ctx.root}\n"
    )
    manifest_path = ctx.results / f"layer_{ctx.layer}_MANIFEST.txt"
    manifest_path.write_text(manifest, encoding="utf-8")
    print(f"wrote {manifest_path}:\n{manifest}")


def stage_verify(ctx: Ctx) -> None:
    run_cmd(ctx, ["activations.py", "verify"])


def stage_dump(ctx: Ctx) -> None:
    acts_path = ctx.data / "acts.f16"
    if acts_path.exists():
        print(f"{acts_path} already exists ({acts_path.stat().st_size} bytes); skipping dump")
    else:
        run_cmd(
            ctx,
            ["activations.py", "dump", "--n-tokens", str(ctx.n_tokens), "--batch-size", "16"],
        )
    run_cmd(ctx, ["sae_stats.py", "--chunk", "2048"])


def stage_prompts(ctx: Ctx) -> None:
    prompts_path = ctx.data / "prompts.json"
    if prompts_path.exists():
        print(f"{prompts_path} already exists; skipping prompts stage")
        return
    run_cmd(ctx, ["activations.py", "prompts", "--n-prompts", "40", "--prompt-len", "8"])


def stage_select(ctx: Ctx) -> None:
    run_cmd(ctx, ["features.py", "select", "--seed", "0"])


def stage_train_dir(ctx: Ctx) -> None:
    run_cmd(
        ctx,
        [
            "train_direction.py", "train",
            "--rank", "64",
            "--gamma", "1.0",
            "--lr", "3e-3",
            "--steps", "2000",
            "--name", f"dir_L{ctx.layer}",
        ],
    )


def stage_anatomy(ctx: Ctx) -> None:
    run_cmd(ctx, ["anatomy.py", "--ckpt", f"dir_L{ctx.layer}"])

    src_shared = ctx.checkpoints / "shared_direction.pt"
    dst_shared = ctx.checkpoints / f"shared_direction_L{ctx.layer}.pt"
    if src_shared.exists():
        shutil.copy2(src_shared, dst_shared)
        print(f"copied {src_shared} -> {dst_shared}")
    else:
        print(f"WARNING: {src_shared} missing after anatomy.py; cannot snapshot it for this layer")

    src_summary = ctx.results / "anatomy_summary.json"
    dst_summary = ctx.results / f"anatomy_summary_L{ctx.layer}.json"
    if src_summary.exists():
        shutil.copy2(src_summary, dst_summary)
        print(f"copied {src_summary} -> {dst_summary}")
    else:
        print(f"WARNING: {src_summary} missing after anatomy.py; cannot snapshot it for this layer")

    l7_path = ctx.checkpoints / "shared_direction_L7.pt"
    out: dict = {"layer": ctx.layer}
    if dst_shared.exists() and l7_path.exists():
        import torch

        d_new = torch.load(dst_shared, map_location="cpu")["d_bar"].float()
        d_l7 = torch.load(l7_path, map_location="cpu")["d_bar"].float()
        cos = float(torch.nn.functional.cosine_similarity(d_new, d_l7, dim=0))
        out["cosine_dbar_vs_L7"] = cos
        print(f"cos(d_bar[L{ctx.layer}], d_bar[L7]) = {cos:.6f}")
    else:
        missing = []
        if not dst_shared.exists():
            missing.append(str(dst_shared))
        if not l7_path.exists():
            missing.append(str(l7_path))
        out["cosine_dbar_vs_L7"] = None
        out["note"] = "cannot compute: missing " + ", ".join(missing)
        print(f"WARNING: {out['note']}")

    out_path = ctx.results / f"layer_{ctx.layer}_dbar_vs_L7.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


def stage_gen(ctx: Ctx) -> None:
    L = ctx.layer
    run_cmd(
        ctx,
        [
            "generate.py",
            "--split", "test",
            "--arms", "naive,dirfix,shared",
            "--c-grid", "0,0.5,1.0,1.5,2.0,3.0",
            "--n-prompts", "30",
            "--set", f"direction=dir_L{L}",
            "--out", f"gen_layer{L}.jsonl",
        ],
    )
    run_cmd(
        ctx,
        [
            "metrics.py",
            "--gen", f"gen_layer{L}.jsonl",
            "--out", f"scored_layer{L}.csv",
            "--split", "test",
            "--stages", "ppl,keyword,sae,dist",
            "--shuffle-frac", "0.15",
        ],
    )
    run_cmd(
        ctx,
        ["paired_at_strength.py", "--scored", f"scored_layer{L}.csv", "--out", f"paired_layer{L}.csv"],
    )
    run_cmd(
        ctx,
        [
            "control_specificity.py",
            "--gen", f"gen_layer{L}.jsonl",
            "--split", "test",
            "--out", f"control_specificity_layer{L}.csv",
        ],
    )


def stage_note(ctx: Ctx) -> None:
    import pandas as pd

    L = ctx.layer
    lines = [f"# Layer {L}: does the same shared direction correction exist, and does it match layer 7?", ""]

    manifest_path = ctx.results / f"layer_{L}_MANIFEST.txt"
    lines.append("## manifest")
    if manifest_path.exists():
        lines.append("```")
        lines.append(manifest_path.read_text(encoding="utf-8").strip())
        lines.append("```")
    else:
        lines.append(f"missing `{manifest_path.name}` (prep stage did not run)")
    lines.append("")

    summary_path = ctx.results / f"anatomy_summary_L{L}.json"
    lines.append("## anatomy of the learned correction on this layer")
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        vs = s.get("delta_cloud_variance_share", {})
        lines.append(f"- pairwise cosine between per-feature corrections: mean={s.get('mean_pairwise_cos')} median={s.get('median_pairwise_cos')}")
        lines.append(f"- mean correction norm / mean direction norm: {s.get('mean_delta_norm_over_mean_norm')}")
        lines.append(f"- correction-cloud variance share: top1={vs.get('top1')} top5={vs.get('top5')} top20={vs.get('top20')}")
        lines.append(f"- M singular values (top 10): {s.get('M_singular_values_top10')}")
        lines.append(f"- M Frobenius norm: {s.get('M_frobenius_norm')}, participation ratio: {s.get('M_participation_ratio')}")
    else:
        lines.append(f"missing `{summary_path.name}` (anatomy stage did not run)")
    lines.append("")

    dbar_path = ctx.results / f"layer_{L}_dbar_vs_L7.json"
    lines.append("## d_bar cosine vs the layer-7 shared direction")
    if dbar_path.exists():
        d = json.loads(dbar_path.read_text(encoding="utf-8"))
        if d.get("cosine_dbar_vs_L7") is not None:
            lines.append(f"cos(d_bar[L{L}], d_bar[L7]) = {d['cosine_dbar_vs_L7']:.6f}")
        else:
            lines.append(f"not computed: {d.get('note')}")
    else:
        lines.append(f"missing `{dbar_path.name}` (anatomy stage did not run)")
    lines.append("")

    scored_path = ctx.results / f"scored_layer{L}.csv"
    lines.append("## arm x strength: fluency and concept metrics (test split, this layer)")
    if scored_path.exists():
        df = pd.read_csv(scored_path)
        g = df.groupby(["arm", "c"])[["logppl", "keyword_hit", "rep4"]].mean().round(4).reset_index()
        lines.append(_df_to_md(g))
    else:
        lines.append(f"missing `{scored_path.name}` (gen stage did not run)")
    lines.append("")

    paired_path = ctx.results / f"paired_layer{L}.csv"
    lines.append("## paired differences vs naive, at matched strength")
    if paired_path.exists():
        df = pd.read_csv(paired_path)
        lines.append(_df_to_md(df.round(4)))
    else:
        lines.append(f"missing `{paired_path.name}` (gen stage did not run)")
    lines.append("")

    out_path = ctx.results / f"layer_{L}_NOTE.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


STAGES = {
    "prep": stage_prep,
    "verify": stage_verify,
    "dump": stage_dump,
    "prompts": stage_prompts,
    "select": stage_select,
    "train_dir": stage_train_dir,
    "anatomy": stage_anatomy,
    "gen": stage_gen,
    "note": stage_note,
}


def run_stage(name: str, fn, ctx: Ctx) -> tuple[str, float]:
    print(f"\n=== stage '{name}' start {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    t0 = time.time()
    try:
        fn(ctx)
        minutes = (time.time() - t0) / 60
        print(f"=== stage '{name}' done in {minutes:.1f} min ===", flush=True)
        return "ok", minutes
    except Exception:
        minutes = (time.time() - t0) / 60
        print(f"!!! stage '{name}' FAILED after {minutes:.1f} min !!!", flush=True)
        traceback.print_exc()
        return "FAILED", minutes


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=None, help="target layer L, 1..11 (moves SAE_ID/HOOK via TLAB_LAYER); falls back to env TLAB_LAYER")
    ap.add_argument("--stages", default=",".join(STAGE_ORDER), help=f"comma-separated subset of: {','.join(STAGE_ORDER)}")
    ap.add_argument("--python", default=sys.executable, help="interpreter used for subprocess calls")
    ap.add_argument("--n-tokens", dest="n_tokens", type=int, default=2_000_000, help="token budget for the dump stage")
    ap.add_argument("--root", default=None, help="repository root (default: parent of this script's directory)")
    ap.add_argument(
        "--i-know-this-overwrites",
        dest="allow_overwrite",
        action="store_true",
        help="allow select/dump/train_dir to run even against what looks like the live repository",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.layer is None:
        # the Kaggle queue runner can only pass --stages, so the layer arrives via the environment
        if not os.environ.get("TLAB_LAYER"):
            raise SystemExit("--layer is required (or set TLAB_LAYER in the environment)")
        args.layer = int(os.environ["TLAB_LAYER"])

    if not (1 <= args.layer <= 11):
        raise SystemExit(f"--layer must be an integer in 1..11, got {args.layer}")

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    if not (root / "src").is_dir():
        raise SystemExit(f"{root / 'src'} does not exist; check --root (got root={root})")

    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in requested if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; choose from {list(STAGE_ORDER)}")

    # First thing, per src/common.py's contract: TLAB_LAYER must be set before common is imported
    # (by this driver or by any subprocess) and before any subprocess is spawned, since children
    # inherit this process's environment.
    os.environ["TLAB_LAYER"] = str(args.layer)
    # So that this driver's own `import common` (prep/anatomy stages) resolves against <root>/src
    # rather than wherever this script happens to live -- required when --root is an isolated copy.
    sys.path.insert(0, str(root / "src"))

    guarded_requested = [s for s in requested if s in GUARDED_STAGES]
    if guarded_requested and not args.allow_overwrite:
        marker = root / "results" / LIVE_REPO_MARKER
        if marker.exists():
            raise SystemExit(
                f"refusing to run stage(s) {guarded_requested}: {marker} exists, which marks "
                f"--root={root} as the live repository (round-2 output), not an isolated copy. "
                "These stages overwrite configs/features.yaml, data/splits.npz, data/acts.f16 or "
                "train a fresh checkpoint from them, which would clobber the frozen layer-7 "
                "pipeline's inputs. Re-run with --root pointing at an isolated copy of the repo, "
                "or pass --i-know-this-overwrites if this is really what you want."
            )

    ctx = Ctx(layer=args.layer, root=root, python=args.python, n_tokens=args.n_tokens)

    outcomes: list[tuple[str, str, float]] = []
    for stage in requested:
        status, minutes = run_stage(stage, STAGES[stage], ctx)
        outcomes.append((stage, status, minutes))

    print("\n=== summary ===")
    for stage, status, minutes in outcomes:
        print(f"{stage:12s} {status:8s} {minutes:6.1f} min")


if __name__ == "__main__":
    main()
