"""Report figures. Each figure answers one question and is built to be read, not decorated."""

from __future__ import annotations

import argparse
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS
from pareto import _front

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
REF = MUTED

ARM_LABEL = {
    "clean": "без стиринга",
    "naive": "наивный стиринг",
    "norm_preserving": "сохранение нормы",
    "denoise_naive": "денойзер (формулировка задания)",
    "cds": "CDS (контрастная коррекция)",
    "mts": "MTS (махаланобис-перенос)",
    "fsr": "FSR (хирургия по признакам)",
    "ctr": "CTR (обучен против downstream)",
}


def style(ax, xlabel: str, ylabel: str, title: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)


def new_fig(w: float, h: float, nrows: int = 1, ncols: int = 1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), facecolor=SURFACE)
    return fig, axes


def save(fig, name: str) -> None:
    out = RESULTS / f"{name}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out.name}")


def _cells(scored: str, concept: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / scored)
    df = df.dropna(subset=["logppl", concept])
    return (
        df.groupby(["arm", "c"])
        .agg(logppl=("logppl", "mean"), concept=(concept, "mean"))
        .reset_index()
    )


def fig_pareto(args) -> None:
    """One panel per repair arm, each against the naive front. Two series per panel keeps
    identity unambiguous and lets the reader check dominance panel by panel."""
    cells = _cells(args.scored, args.concept)
    base = cells[cells["arm"] == "naive"].sort_values("c")
    arms = [a for a in ARM_LABEL if a in set(cells["arm"]) and a not in ("clean", "naive")]
    n = len(arms)
    if n == 0:
        print("skip fig_pareto: no repair arms present")
        return
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = new_fig(4.4 * ncols, 3.6 * nrows, nrows, ncols)
    axes = np.atleast_1d(axes).ravel()

    for ax, arm in zip(axes, arms):
        sub = cells[cells["arm"] == arm].sort_values("c")
        fx, fy = _front(base["logppl"].to_numpy(), base["concept"].to_numpy())
        ax.plot(fx, fy, color=REF, linewidth=2, marker="o", markersize=5, zorder=2, label=ARM_LABEL["naive"])
        gx, gy = _front(sub["logppl"].to_numpy(), sub["concept"].to_numpy())
        ax.plot(gx, gy, color=SERIES[0], linewidth=2, marker="o", markersize=6, zorder=3, label=ARM_LABEL[arm])
        for _, r in sub.iterrows():
            if r["c"] in (1.0, 2.0, 3.0):
                ax.annotate(
                    f"c={r['c']:g}",
                    (r["logppl"], r["concept"]),
                    textcoords="offset points",
                    xytext=(6, -3),
                    fontsize=8,
                    color=MUTED,
                )
        style(ax, "log-перплексия продолжения (ниже — лучше)", "доля продолжений с концептом", ARM_LABEL[arm])
        leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
        for t in leg.get_texts():
            t.set_color(INK2)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    save(fig, "fig_pareto")


def fig_transmission(args) -> None:
    path = RESULTS / "transmission.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    g = df.groupby(["denoiser", "c"])["tau"].agg(["mean", "std"]).reset_index()
    fig, ax = new_fig(6.2, 4.0)
    for i, (name, sub) in enumerate(g.groupby("denoiser")):
        col = SERIES[i % len(SERIES)]
        ax.plot(sub["c"], sub["mean"], color=col, linewidth=2, marker="o", markersize=6, label=name)
        ax.fill_between(
            sub["c"], sub["mean"] - sub["std"], sub["mean"] + sub["std"], color=col, alpha=0.12, linewidth=0
        )
    ax.axhline(1.0, color=AXIS, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("полная передача", (ax.get_xlim()[0], 1.0), textcoords="offset points", xytext=(4, 4), fontsize=8, color=MUTED)
    style(ax, "сила стиринга c (в единицах естественного масштаба признака)", "передача τ", "Сколько стиринга денойзер оставляет")
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    save(fig, "fig_transmission")


def fig_endpoint(args) -> None:
    path = RESULTS / "endpoint_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["arm"].isin(ARM_LABEL)].copy()
    df["label"] = df["arm"].map(ARM_LABEL)
    df = df.sort_values("delta_mean")
    fig, ax = new_fig(7.0, 0.55 * len(df) + 1.6)
    y = np.arange(len(df))
    ax.barh(y, df["delta_mean"], color=SERIES[0], height=0.6, zorder=3)
    ax.errorbar(
        df["delta_mean"], y, xerr=[df["delta_mean"] - df["lo95"], df["hi95"] - df["delta_mean"]],
        fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=4,
    )
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=9, color=INK2)
    for yi, (v, p) in enumerate(zip(df["delta_mean"], df["p_gt_0"])):
        ax.annotate(f"{v:+.3f}  p={p:.2f}", (v, yi), textcoords="offset points", xytext=(6, -3), fontsize=8, color=INK2)
    style(ax, "прирост концепта при выровненной связности", "", "Primary endpoint против наивного стиринга")
    fig.tight_layout()
    save(fig, "fig_endpoint")


def fig_spectral(args) -> None:
    path = RESULTS / "spectral.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    bins = [c for c in df.columns if c.startswith("bin")]
    fig, axes = new_fig(9.0, 3.8, 1, 2)
    for ax, c in zip(np.atleast_1d(axes).ravel(), sorted(df["c"].unique())):
        sub = df[df["c"] == c]
        for i, (arm, g) in enumerate(sub.groupby("arm")):
            frac = g[bins].mean() / g["total_energy"].mean()
            col = REF if arm == "naive" else SERIES[(i - 1) % len(SERIES)]
            ax.plot(range(len(bins)), frac.to_numpy(), color=col, linewidth=2, marker="o", markersize=4, label=ARM_LABEL.get(arm, arm))
        ax.set_yscale("log")
        style(ax, "децили собственных направлений Σ (слева — высокая дисперсия)", "доля энергии возмущения", f"c = {c:g}")
    leg = np.atleast_1d(axes).ravel()[0].legend(frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    save(fig, "fig_spectral")


def fig_surgery(args) -> None:
    path = RESULTS / "surgery_clamp.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    g = df.groupby("c")[["n_over_per_token", "total_excess"]].agg(["mean", "std"])
    fig, axes = new_fig(9.0, 3.8, 1, 2)
    axes = np.atleast_1d(axes).ravel()
    for ax, col, lab in zip(
        axes,
        ["n_over_per_token", "total_excess"],
        ["не-целевых латентов выше потолка, на токен", "суммарное превышение потолка"],
    ):
        m, s = g[(col, "mean")], g[(col, "std")]
        ax.plot(m.index, m.to_numpy(), color=SERIES[0], linewidth=2, marker="o", markersize=6)
        ax.fill_between(m.index, (m - s).to_numpy(), (m + s).to_numpy(), color=SERIES[0], alpha=0.12, linewidth=0)
        style(ax, "сила стиринга c", lab)
    axes[0].set_title("Поломка разрежена: её несёт небольшой набор захваченных признаков", color=INK, fontsize=11, loc="left", pad=8)
    fig.tight_layout()
    save(fig, "fig_surgery")


def fig_predictors(args) -> None:
    ep = RESULTS / "endpoint_per_feature.csv"
    if not ep.exists():
        return
    import yaml

    from common import ROOT

    tbl = pd.read_csv(ep)
    feats = yaml.safe_load((ROOT / "configs" / "features.yaml").read_text(encoding="utf-8"))["test"]
    geo = pd.DataFrame(
        [{"feature": int(r["index"]), "var_along": r["var_along"], "kappa_tilt": r["kappa_tilt"]} for r in feats]
    )
    base = tbl[tbl["arm"] == "naive"][["feature", "concept_at_budget"]].rename(columns={"concept_at_budget": "base"})
    arm = args.gain_arm
    m = tbl[tbl["arm"] == arm].merge(base, on="feature").merge(geo, on="feature").dropna()
    if m.empty:
        return
    m["gain"] = m["concept_at_budget"] - m["base"]
    fig, axes = new_fig(9.0, 3.8, 1, 2)
    for ax, pred, lab in zip(
        np.atleast_1d(axes).ravel(),
        ["kappa_tilt", "var_along"],
        ["наклон κ_tilt = ‖P⊥Σv̂‖ / v̂ᵀΣv̂", "дисперсия вдоль направления v̂ᵀΣv̂"],
    ):
        ax.scatter(m[pred], m["gain"], s=64, color=SERIES[0], zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.axhline(0, color=AXIS, linewidth=1)
        style(ax, lab, "прирост концепта при равной связности")
    np.atleast_1d(axes).ravel()[0].set_title(f"Предикторы выигрыша, плечо {ARM_LABEL.get(arm, arm)}", color=INK, fontsize=11, loc="left", pad=8)
    fig.tight_layout()
    save(fig, "fig_predictors")


def fig_axes_sanity(args) -> None:
    """Both axes, plotted against strength, including the non-monotone concept curve and the
    guard metric that catches the model ignoring the prompt."""
    df = pd.read_csv(RESULTS / args.scored)
    sub = df[df["arm"] == "naive"]
    cols = [c for c in ("keyword_hit", "sae_act", "judge") if c in sub.columns]
    g = sub.groupby("c")[["logppl", "prompt_dependence", "rep4"] + cols].mean().reset_index()
    fig, axes = new_fig(9.0, 3.8, 1, 2)
    axes = np.atleast_1d(axes).ravel()
    metric_label = {
        "keyword_hit": "лексический признак концепта",
        "sae_act": "активация целевого признака SAE",
        "judge": "оценка судьи",
    }
    ax = axes[0]
    for i, c in enumerate(cols):
        v = g[c] / max(g[c].max(), 1e-9)
        ax.plot(
            g["c"], v, color=SERIES[i % len(SERIES)], linewidth=2, marker="o", markersize=6,
            label=metric_label.get(c, c),
        )
    style(ax, "сила стиринга c", "концепт (нормировано на максимум)", "Концепт немонотонен по силе")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK2)
    ax = axes[1]
    ax.plot(g["c"], g["logppl"], color=SERIES[0], linewidth=2, marker="o", markersize=6, label="log-PPL")
    ax.plot(g["c"], g["prompt_dependence"], color=SERIES[1], linewidth=2, marker="o", markersize=6, label="prompt dependence")
    ax.axhline(0, color=AXIS, linewidth=1)
    style(ax, "сила стиринга c", "нат", "Связность падает, промпт перестаёт читаться")
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    save(fig, "fig_axes_sanity")


def fig_agreement(args) -> None:
    df = pd.read_csv(RESULTS / args.scored)
    cols = [c for c in ("keyword_hit", "sae_act", "judge") if c in df.columns]
    if len(cols) < 2:
        return
    g = df.groupby(["arm", "feature", "c"])[cols].mean().reset_index()
    from scipy.stats import spearmanr

    pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]
    fig, axes = new_fig(4.4 * len(pairs), 3.8, 1, len(pairs))
    axes = np.atleast_1d(axes).ravel()
    rows = []
    for ax, (a, b) in zip(axes, pairs):
        rho, p = spearmanr(g[a], g[b])
        rows.append({"x": a, "y": b, "spearman": rho, "p": p, "n": len(g)})
        ax.scatter(g[a], g[b], s=18, color=SERIES[0], alpha=0.5, zorder=3, linewidth=0)
        style(ax, a, b, f"ρ_s = {rho:.2f}")
    pd.DataFrame(rows).to_csv(RESULTS / "metric_agreement.csv", index=False)
    fig.tight_layout()
    save(fig, "fig_agreement")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="scored_test.csv")
    ap.add_argument("--concept", default="keyword_hit")
    ap.add_argument("--gain-arm", dest="gain_arm", default="cds")
    a = ap.parse_args()
    for fn in (
        fig_pareto,
        fig_axes_sanity,
        fig_agreement,
        fig_transmission,
        fig_endpoint,
        fig_spectral,
        fig_surgery,
        fig_predictors,
    ):
        try:
            fn(a)
        except FileNotFoundError as e:
            print(f"skip {fn.__name__}: {e}")
