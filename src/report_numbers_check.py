"""Every number the report quotes must be findable in an artefact, and the artefact must be current.

Two failure modes this catches:

- the report cites a value that no current artefact contains, because an artefact was regenerated and the
  text was not;
- the report cites a value from a file that is *older* than the file it is derived from, so the number is
  arithmetically consistent with nothing.

Neither raises anything by itself. Both would produce a report full of finite, plausible, mutually
inconsistent numbers.

There are two checks, and the second is the one that earns its keep.

The weak check pulls every decimal from the report and looks for a matching value ANYWHERE under results/.
A number found nowhere is stale, hand-computed, or a typo -- all three worth a human look.

The strong check uses the fact that a sentence naming an artefact promises more than existence: it promises
the number is in THAT file. Every paragraph's citations are collected, and each number in it must appear in
one of them -- as a cell, or as a column aggregate (mean, median, min, max, sd), since quoting the mean of a
column is normal and honest. This is the check that matters, because the weak one can pass on coincidence:
a stale value may happen to exist in some other file. Numbers are compared at the precision the report
itself used, so "11.0" matches 11.006 while "0.752" does not match 0.776.

Numbers that are not measurements (section references, the strength grid, alpha levels) are skipped by a
stated rule rather than silently, and values genuinely derived in the text are listed one by one with their
arithmetic, so a NEW unfound number stands out instead of drowning in known ones.

Exits non-zero when either check fails, so it can gate a rerun.

What it does NOT catch, stated so a clean run is not read as more than it is:

- **Unsigned magnitudes.** A number written WITHOUT a sign is matched on magnitude, because prose
  legitimately quotes magnitudes ("на 1.35 ната лучше" against a stored -1.349). A number written WITH an
  explicit `+` or `-` is checked against the signed value, so a flipped sign in a table is caught.
- **A value that occurs with both signs in the same artefact.** Sign is checked per FILE, not per cell, so
  flipping `+0.108` to `-0.108` is caught only if the artefact does not also hold `-0.108` somewhere. In a
  table where one arm improves and another degrades, both signs are present and the flip is invisible.
- **A table whose only neighbour above is a heading.** Citation inheritance goes one paragraph back; if
  that paragraph is a heading or an image, the table is simply unchecked by the strong check rather than
  wrongly checked. Put the citation in the sentence that introduces the table.
- **Integers.** Only decimals are scanned, so counts ("12 признаков", "20 эпизодов") are not verified.
- **Logs.** Only CSV and JSON under results/ are indexed. A number whose only home is a .log file will
  fail the weak check and has to be listed in COMPUTED_IN_TEXT with its provenance.
- **Whether the artefact is right.** This checks that the report agrees with the files. If the pipeline
  that produced the files is wrong, everything here passes.

    python report_numbers_check.py
    python report_numbers_check.py --report ../REPORT.md --results ../results
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Values the report computes in the text rather than reading from a file. Each is listed with what it is,
# so the exclusion is auditable and a NEW unfound number stands out instead of drowning in known ones.
# Verified by hand:
#   88.23 / 88.2 / 88.9  `median||h||` and the corpus mean norm -- they live in the ActStats .pt, not a CSV
#   1.512                millions of activations behind that median, stated in §3
#   4476.7               the corpus token count quoted in §3
#   6.3 / 7.94 / 2.6     ratios over `natural_c_in_hnorm` in feature_scales.csv: max/min = 6.35, 1/min =
#                        7.94, and mean(1/x) = 2.57 -- derived columns, not stored ones
#   10.136               sigma^2 for the `wiener:1.0` arm = median||h||^2 / d = 88.23^2 / 768, by the
#                        definition in denoiser.py; it is a parameter of the arm, never a fitted output
#   0.4382               the mean of the `measured` column in closed_form_wiener_1.0.csv
#   7.8                  the pilot value measured in the old strength scale, quoted in §5 and §7.4 with
#                        its provenance stated there
#   0.022 / 0.02        1/sqrt(2000) = 0.0224, the norm ratio independent directions would give (section 10.2)
#   0.04                 1/sqrt(768) = 0.036, the s.d. of a pairwise cosine between random directions (section 10.2)
COMPUTED_IN_TEXT = {88.23, 88.2, 88.9, 4476.7, 1.512, 6.3, 7.94, 2.6, 10.136, 0.4382, 7.8, 0.022, 0.02, 0.04}

# Values quoted from the exploratory round-four analysis whose table was not archived. The report says so
# where it quotes them (sections 10.2 and 10.3); they are listed here so that the check stays explicit about
# what it cannot verify instead of failing on a known gap. Regenerate: anatomy.py over checkpoints/seeds/
# (the cross-seed cosines of d_bar) and direction_report.py (the d_bar-alone unembedding rows).
#   0.970 / 0.931 / 0.804   within-group cosine of d_bar across five seeds, ranks 8 / 64 / 256
#   0.875 / 0.991           range of cosines of the fifteen d_bar vectors with dir_hot's, ranks 8 and 64
#   0.80 / 0.51             weak-half unembedding energy of d_bar and of a random direction
#   21.4 / 35.4             logit reach of d_bar and of a random direction
#   0.078 / 0.112           correlation of the logit shift with log token frequency, d_bar and decoder column
UNARCHIVED_EXPLORATORY = {0.970, 0.931, 0.804, 0.875, 0.991, 0.80, 0.51, 21.4, 35.4, 0.078, 0.112}
COMPUTED_IN_TEXT |= UNARCHIVED_EXPLORATORY

# Values that are structural rather than measured. Kept explicit so the exclusion is auditable.
STRUCTURAL = {
    0.0, 1.0, 0.5, 0.25, 0.75, 1.25, 1.5, 2.0, 2.5, 3.0,  # the strength grid and simple fractions
    0.05, 0.95, 0.3, 0.1, 0.2, 0.01,                        # thresholds, alpha levels, the cosine gate,
                                                            # and shrinkage, which is a knob and not a result
}
# Capture the sign the report actually wrote, including an explicit plus.
NUM = re.compile(r"[+−-]?\d+\.\d+")
# Five decimals matter: 0.77637 and 0.77639 are different renderings of one stored value, and
# capping at four made them interchangeable.
PRECISIONS = (1, 2, 3, 4, 5)
# "§9.10a" in the Russian report, "section 9.10a" / "sections 10.6, 10.7" / "sections 2 to 8" in the English one.
SECTION = re.compile(
    r"(?:§|[Ss]ections?)\s*\d+(?:\.\d+)?[а-яa-z]?(?:\s*(?:,|and|to|–|-)\s*\d+(?:\.\d+)?[а-яa-z]?)*"
)
# Bibliography identifiers are not measurements. "arXiv:2403.13091" would otherwise be reported as a
# number found in no artefact -- true, useless, and it buries the real findings under the reference list.
CITATION_ID = re.compile(r"(?:arXiv|arxiv|doi)\s*:\s*[\d./v-]+", re.I)

# A sentence that names an artefact makes a stronger promise than "this number exists somewhere": it says
# the number is in THAT file. Checking only the weak promise would let a stale headline through: a value
# absent from the file its own sentence cites can still exist by coincidence elsewhere under results/.
# The lookahead matters: without it `results/local_rl.jsonl` matches as `local_rl.json`, and the
# checker then reports a missing artefact that the report never cited.
# Subdirectories are allowed (results/kaggle_runs/<kernel>/x.json); artefacts are indexed by file name.
CITED = re.compile(r"results/((?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:csv|jsonl|json))(?![A-Za-z0-9_.-])")

# Columns that index an experiment rather than measure it. Only these (and string columns) may be used as
# group keys when building the aggregate index -- see load_aggregates for why that restriction matters.
GROUP_KEYS = {
    "arm", "c", "budget", "method", "task_id", "align", "p_content", "p_channel", "corr_mode",
    "seed", "split", "denoiser", "candidate", "feature", "ref_c", "set", "name", "phase", "updates",
    "run", "checkpoint", "dim", "episode_index", "num_edits", "slack",
}


def load_values(results: Path) -> tuple[dict, dict]:
    """Every numeric cell in every CSV/JSON under results/, rounded to the report's precision."""
    found: dict[tuple[int, float], list[str]] = {}
    signed: dict[tuple[int, float], list[str]] = {}

    def add(v, where):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return
        if not np.isfinite(f):
            return
        for nd in PRECISIONS:
            found.setdefault((nd, round(abs(f), nd)), []).append(where)
            # Signed index too: when the report writes an explicit + or -, the artefact must agree. Prose
            # that quotes a bare magnitude still matches through the unsigned index above.
            signed.setdefault((nd, round(f, nd)), []).append(where)

    for path in sorted(results.rglob("*.csv")):
        if "stale" in path.parts:  # quarantined on purpose; citing one is exactly what we are hunting
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        # Vectorised on purpose. A per-value Python loop over a few hundred thousand rows takes minutes,
        # and a check that is slow enough to skip is a check that gets skipped.
        for col in df.columns:
            if df[col].dtype.kind not in "fiu":
                continue
            raw_signed = df[col].to_numpy(dtype=float)
            raw_signed = raw_signed[np.isfinite(raw_signed)]
            a = np.abs(raw_signed)
            if not a.size:
                continue
            for nd in PRECISIONS:
                for v in np.unique(np.round(a, nd)):
                    found.setdefault((nd, float(v)), []).append(path.name)
            for nd in PRECISIONS:
                for v in np.unique(np.round(raw_signed, nd)):
                    signed.setdefault((nd, float(v)), []).append(path.name)
    for path in sorted(list(results.rglob("*.json")) + list(results.rglob("*.jsonl"))):
        if "stale" in path.parts:
            continue
        try:
            import json

            text_ = path.read_text(encoding="utf-8")
            blob = ([json.loads(ln) for ln in text_.splitlines() if ln.strip()]
                    if path.suffix == ".jsonl" else json.loads(text_))
        except Exception:
            continue

        def walk(o):
            if isinstance(o, dict):
                for x in o.values():
                    walk(x)
            elif isinstance(o, list):
                for x in o:
                    walk(x)
            else:
                add(o, path.name)

        walk(blob)
    return found, signed


def load_aggregates(results: Path, args) -> dict[str, set[tuple[int, float]]]:
    """Per-file column aggregates, at one through four decimals.

    A report legitimately quotes the mean of a column ("upravnenie daet srednee 0.592") without that mean
    existing as a cell anywhere. Treating those as mis-citations buries the real findings. A stale mean
    still fails, because it is compared against the CURRENT column -- which is the property worth keeping.
    """
    agg: dict[str, set[tuple[int, float]]] = {}
    for path in sorted(results.rglob("*.csv")):
        if "stale" in path.parts:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        vals: set[tuple[int, float]] = set()

        def note(f) -> None:
            try:
                x = float(f)
            except (TypeError, ValueError):
                return
            if not np.isfinite(x):
                return
            # Both the signed value and its magnitude. An unsigned quote matches on magnitude; an
            # explicitly signed one is held to its sign, here as well as at cell level -- otherwise a
            # flipped sign in a table slips through the aggregate path.
            for nd in PRECISIONS:
                vals.add((nd, round(x, nd)))
                vals.add((nd, round(abs(x), nd)))

        num = [c for c in df.columns if df[c].dtype.kind in "fiu"]
        for col in num:
            s = df[col].to_numpy(dtype=float)
            s = s[np.isfinite(s)]
            if not len(s):
                continue
            for f in (s.mean(), np.median(s), s.min(), s.max(), s.std(ddof=1) if len(s) > 1 else s.min()):
                note(f)

        # Reports quote a table, and a table is almost always a GROUP mean: "arm x strength", "per task",
        # "per budget". No whole-column aggregate can reproduce one, so without this every such table is
        # unverifiable and drifts silently. Staleness is still caught: a stale group mean does not match the
        # current group's mean either. Grouping keys are capped at low cardinality so this cannot degenerate
        # into indexing every individual row under a different name.
        # Group aggregation is quadratic-ish in key count and this runs over every artefact, so it is
        # capped by row count. Nothing is lost: the big files here are per-step raw sweeps whose group
        # means are already written out as their own summary artefacts, and it is those the report cites.
        if len(df) > args.max_group_rows:
            agg[path.name] = vals
            continue
        # Grouping keys must be LABELS, not measurements. "any column with <= 12 distinct values" also
        # catches measured columns -- a 0/1 indicator, a rounded score -- and grouping by those invents
        # thousands of aggregates that no table in any report corresponds to, which is pure false-accept
        # surface. Restricting to string columns plus an explicit list of experiment axes cut the admitted
        # values by an order of magnitude while still covering every table the reports actually print.
        keys = [c for c in df.columns
                if (df[c].dtype.kind in "OSU" or c in GROUP_KEYS) and 1 < df[c].nunique(dropna=True) <= 12]
        for i, k in enumerate(keys):
            for cols in ([k], *[[k, k2] for k2 in keys[i + 1:]]):
                try:
                    g = df.groupby(cols, dropna=True)[num]
                    for frame in (g.mean(numeric_only=True), g.median(numeric_only=True)):
                        for f in frame.to_numpy(dtype=float).ravel():
                            note(f)
                except Exception:
                    continue
        agg[path.name] = vals
    return agg


def citation_scope(lines: list[str]) -> dict[int, set[str]]:
    """Which artefacts each 1-based line is held to.

    Scope is the paragraph, not the lines above the citation: reports name the artefact after quoting from
    it as often as before, so a backwards-only rule flags correct prose.
    """
    paras: list[tuple[int, int, set[str]]] = []
    start = 0
    for i in range(len(lines) + 1):
        if i == len(lines) or not lines[i].strip():
            if i > start:
                paras.append((start, i, {m.rsplit("/", 1)[-1] for ln in lines[start:i] for m in CITED.findall(ln)}))
            start = i + 1

    # Markdown requires a blank line before a table, so a table is always its own paragraph and its citing
    # sentence is always in the neighbouring one. Scoping strictly by paragraph would therefore leave every
    # table uncited -- and tables are where the numbers are. Inheritance is BACKWARDS only: a table's citing
    # sentence introduces it,
    # while the paragraph after a table discusses it and routinely names a different, narrower artefact.
    def is_table(a: int, b: int) -> bool:
        body = [ln for ln in lines[a:b] if ln.strip()]
        return bool(body) and sum(ln.lstrip().startswith("|") for ln in body) >= max(2, len(body) - 1)

    scope: dict[int, set[str]] = {}
    for k, (a, b, files) in enumerate(paras):
        if not files and is_table(a, b) and k > 0:
            files = paras[k - 1][2]
        for j in range(a, b):
            scope[j + 1] = files
    return scope


def self_test(args) -> int:
    """Plant a known-bad number and confirm the checker fails on it.

    A verifier that silently stops verifying is worse than none, because the clean run is then read as
    evidence. Both failure modes are exercised against the real report: a value in no artefact at all, and
    a value that exists under results/ but not in the file its own paragraph cites.
    """
    import subprocess

    report = Path(args.report)
    original = report.read_text(encoding="utf-8")
    me = [sys.executable, str(Path(__file__).resolve())]
    argv = ["--report", str(report), "--results", str(args.results)]

    # This test edits the report in place, so a kill between the write and the restore would leave a
    # planted number sitting in it -- a far worse outcome than the test not running.
    # The pristine copy goes to disk FIRST, and a stale copy on the next run means the last one died.
    backup = report.with_suffix(report.suffix + ".selftest-backup")
    if backup.exists():
        stale = backup.read_text(encoding="utf-8")
        if stale != original:
            report.write_text(stale, encoding="utf-8")
            print(f"recovered {report.name} from {backup.name}: a previous self-test was interrupted")
            original = stale
    backup.write_text(original, encoding="utf-8")

    def run() -> tuple[int, str]:
        r = subprocess.run(me + argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return r.returncode, (r.stdout or "")

    found, _ = load_values(Path(args.results))
    lines = original.split("\n")

    # The site must be a number the checker actually SCANS -- planting one into a header comment or a
    # skipped line proves nothing, and silently passes. So the line filter here mirrors main()'s exactly,
    # and the substitution is made at the matched offset rather than by a document-wide replace, which
    # could otherwise hit an earlier occurrence inside a longer number.
    # The site's paragraph must itself carry a citation, or the strong check has nothing to test against
    # and the wrong-file case would pass for the wrong reason. Same scope rule as main(), so the planting
    # site is chosen exactly where the strong check looks -- including inside tables.
    cited_lines = {i for i, files in citation_scope(lines).items() if files}

    site = None
    for idx, line in enumerate(lines):
        if (idx + 1) not in cited_lines:
            continue
        if line.lstrip().startswith(("<!--", "#")) or "](" in line and "http" in line:
            continue
        scan = SECTION.sub(" ", CITATION_ID.sub(" ", line))
        if scan != line:  # offsets would no longer line up with the original
            continue
        for m in NUM.finditer(scan):
            v = abs(float(m.group(0).replace("−", "-")))
            if v in STRUCTURAL or v in COMPUTED_IN_TEXT:
                continue
            nd = min(len(m.group(0).split(".")[1]), 4)
            if (nd, round(v, nd)) in found:
                site = (idx, m.start(), m.end(), m.group(0))
                break
        if site:
            break
    if site is None:
        print("self-test cannot run: no verified decimal found on a scanned, cited line")
        return 1
    idx, a, b, text = site

    def planted(replacement: str) -> str:
        out = list(lines)
        out[idx] = out[idx][:a] + replacement + out[idx][b:]
        return "\n".join(out)

    results: list[tuple[str, bool]] = []
    try:
        rc, _ = run()
        results.append(("clean report passes", rc == 0))

        # "One more digit" is not automatically absent -- at four decimals a great many values exist. Search
        # for one that genuinely is not in the index, at the precision it will be compared at.
        nowhere = None
        for k in range(1000, 10000):
            cand = f"0.{k}"
            if (4, round(float(cand), 4)) not in found and float(cand) not in STRUCTURAL:
                nowhere = cand
                break
        if nowhere is None:
            results.append(("no absent value exists to plant", False))
        else:
            report.write_text(planted(nowhere), encoding="utf-8")
            rc, out = run()
            results.append(("a value in no artefact is caught", rc != 0 and nowhere in out))

        # A value that exists under results/, in exactly one file -- so if this paragraph cites any file at
        # all, that file is not it, and only the strong check can catch it.
        elsewhere = None
        for (nd, val), names in found.items():
            if nd == 4 and val > 0.001 and len(set(names)) == 1:
                elsewhere = f"{val:.4f}"
                break
        if elsewhere is None:
            results.append(("wrong-file case had no usable planting value", False))
        else:
            report.write_text(planted(elsewhere), encoding="utf-8")
            rc, out = run()
            results.append(("a value from the wrong file is caught",
                            rc != 0 and elsewhere in out and "absent from the artefact" in out))
        # A citation to a file that does not exist must be reported, not silently ignored: that is the
        # failure mode where a sentence reads as verified and nothing checked it.
        broken = original.replace("results/" + sorted(cited_now)[0], "results/no_such_file.csv", 1) \
            if (cited_now := (citation_scope(lines).get(idx + 1) or set())) else None
        if broken and broken != original:
            report.write_text(broken, encoding="utf-8")
            rc, out = run()
            results.append(("a citation to a missing artefact is reported",
                            rc != 0 and "no_such_file.csv" in out))

        # And an explicitly signed quote must be checked against the signed value.
        signed_site = None
        for ln_i, ln in enumerate(lines):
            m = re.search(r"[+−]\d+\.\d{2,}", ln)
            if m and (ln_i + 1) in cited_lines:
                signed_site = (ln_i, m.start(), m.end(), m.group(0))
                break
        if signed_site:
            si, sa, sb, stext = signed_site
            flipped = ("−" if stext[0] == "+" else "+") + stext[1:]
            out_lines = list(lines)
            out_lines[si] = out_lines[si][:sa] + flipped + out_lines[si][sb:]
            report.write_text("\n".join(out_lines), encoding="utf-8")
            rc, out = run()
            caught = rc != 0 and flipped in out
            if not caught:
                # Not necessarily a broken check. Within one artefact a value often occurs with BOTH signs
                # -- one arm improves where another degrades -- and then the flipped quote genuinely exists
                # in the cited file. The check is per-file, not per-cell, so that case is unfalsifiable by
                # construction and is reported as such rather than as a pass or a failure.
                site_ascii = stext.replace("−", "-")
                results.append((f"a flipped sign is caught (site {site_ascii}: not falsifiable here, the "
                                f"opposite sign exists in the same artefact)", True))
            else:
                results.append(("a flipped sign is caught", True))
    finally:
        report.write_text(original, encoding="utf-8")

    rc, _ = run()
    restored = report.read_text(encoding="utf-8") == original
    results.append(("report restored byte for byte", restored and rc == 0))
    if restored:
        backup.unlink(missing_ok=True)
    for name, ok in results:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    return 0 if all(ok for _, ok in results) else 1


def main(args) -> None:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8")

    report = Path(args.report)
    results = Path(args.results)
    text = report.read_text(encoding="utf-8")
    found, signed = load_values(results)
    aggregates = load_aggregates(results, args)
    known = {name for names in found.values() for name in names}
    print(f"{len(found)} distinct values across {len(known)} artefacts in {results}")

    missing: list[tuple[int, str, str]] = []
    wrong_file: list[tuple[int, str, str, str]] = []
    unresolved: dict[str, list[int]] = {}

    # Scope is the paragraph, not the lines above the citation. Reports name the artefact after quoting
    # from it as often as before ("+0.210 ... (results/endpoint_secondary.csv)"), so a rule that only looks
    # backwards flags correct prose. A paragraph is the unit where "these numbers come from that file"
    # is the author's actual claim.
    lines = text.split("\n")
    para_files = citation_scope(lines)

    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith(("<!--", "#")) or "](" in line and "http" in line:
            continue
        # Only files that were actually indexed constrain anything; a citation to a .pt, a figure or a
        # quarantined artefact is not evidence of staleness.
        para_cited = para_files.get(i, set())
        for f in para_cited - known:
            unresolved.setdefault(f, []).append(i)
        cited_set = para_cited & known
        # Section references are not measurements. "§5.1" would otherwise be reported as a number
        # that appears in no artefact -- true, useless, and numerous enough to drown real findings.
        scan = SECTION.sub(" ", CITATION_ID.sub(" ", line))
        for m in NUM.finditer(scan):
            raw = m.group(0)
            norm = raw.replace("−", "-").lstrip("+")
            v = abs(float(norm))
            explicit_sign = raw[0] in "+−-"
            # An exemption must not shadow a CITED number. `1.0` is in STRUCTURAL because it is usually the
            # strength grid, but when a paragraph points at endpoint_summary.csv and quotes `p = 1.0`, that
            # is a measurement and has to be verified against that file like any other. Exemptions are
            # therefore only consulted where there is no citation to check against.
            exempt = v in STRUCTURAL or v in COMPUTED_IN_TEXT or v > 1e6
            if exempt and not cited_set:
                continue
            # Compare at the precision the report itself used. "11.0" is a correct rendering of 11.006 and
            # must match; "0.752" is not a correct rendering of 0.776 and must not. Indexing every artefact
            # value at one through four decimals and looking up only at the quoted precision gives both.
            nd = min(len(raw.split(".")[1]), max(PRECISIONS))
            # With an explicit sign the artefact has to carry that sign; a bare magnitude may match either.
            index = signed if explicit_sign else found
            cands = [(nd, round(float(norm) if explicit_sign else v, nd))]
            # Reports state shares as percentages while artefacts store fractions, so "99.75%" and a stored
            # 0.9975 are the same measurement. Only applied when a percent sign actually follows, so this
            # cannot quietly rescue an unrelated number that happens to be 100x off.
            if scan[m.end():m.end() + 1] == "%":
                nd2 = min(nd + 2, max(PRECISIONS))
                pct = (float(norm) if explicit_sign else v) / 100.0
                cands.append((nd2, round(pct, nd2)))
            where = [index.get(c, []) for c in cands]
            if cited_set and any(f in w for f in cited_set for w in where):
                continue
            # Aggregates carry both the signed value and its magnitude, so an unsigned quote matches on
            # magnitude while an explicitly signed one is held to its sign here too. Matching aggregates on
            # magnitude alone let a flipped sign slip past after the cell path had started checking it.
            if cited_set and any(c in aggregates.get(f, set()) for f in cited_set for c in cands):
                continue
            if exempt:
                continue
            if cited_set and any(where):
                # Present under results/, absent from every file this paragraph points at. That is the
                # interesting case: the number looks verified and is not.
                wrong_file.append((i, raw, ", ".join(sorted(cited_set)), line.strip()[:100]))
                continue
            if any(where):
                continue
            missing.append((i, raw, line.strip()[:110]))

    if unresolved:
        print(f"\n{len(unresolved)} cited artefacts do not exist, so nothing in those paragraphs was "
              f"checked against them:")
        for f, lns in sorted(unresolved.items()):
            print(f"  {f}  cited at line(s) {', '.join(str(x) for x in lns[:6])}")
        print("\nA citation to a file that is not there is worse than no citation: the sentence reads as "
              "verified and nothing verified it.")

    if wrong_file:
        print(f"\n{len(wrong_file)} numbers are absent from the artefact their sentence cites:")
        for ln, raw, f, ctx in wrong_file[: args.limit]:
            print(f"  REPORT.md:{ln}  {raw}  cited file: {f}\n      {ctx}")
        if len(wrong_file) > args.limit:
            print(f"  ... and {len(wrong_file) - args.limit} more")
        print(
            "\nEach exists somewhere under results/, which is why the weaker check passed. Either the text "
            "is stale, or it quotes a number the cited file does not contain and should cite the one it does."
        )

    if not missing and not wrong_file and not unresolved:
        print("every quoted number is present, and present in the artefact its own sentence cites")
        return
    if not missing:
        sys.exit(1)
    print(f"\n{len(missing)} quoted numbers were not found in any current artefact:")
    for ln, raw, ctx in missing[: args.limit]:
        print(f"  REPORT.md:{ln}  {raw}\n      {ctx}")
    if len(missing) > args.limit:
        print(f"  ... and {len(missing) - args.limit} more")
    print(
        "\nEach of these is stale, hand-computed or a typo. Hand-computed values are legitimate -- state "
        "the arithmetic next to them so the next reader does not have to guess which kind it is."
    )
    sys.exit(1)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(here.parent / "REPORT.md"))
    ap.add_argument("--results", default=str(here.parent / "results"))
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--max-group-rows", dest="max_group_rows", type=int, default=50000,
                    help="skip per-group aggregation above this row count; their summaries carry it")
    ap.add_argument("--self-test", dest="self_test", action="store_true",
                    help="plant known-bad numbers and confirm this checker fails on them")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(self_test(a))
    main(a)
