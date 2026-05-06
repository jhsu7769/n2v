"""Table 1 — Exp 1 verdict matrix + % solved.

Columns: correct UNSAT, correct SAT, false UNSAT, false SAT,
indeterminate (TIMEOUT + ERROR + NOT_APPLICABLE), % solved, mean
wall-clock per instance.

Sound verifiers (Exp 1): αβ-CROWN, NeuralSAT, PyRAT, CORA. Marabou is
intentionally dropped per the 2026-04 paper revision.

Probabilistic baselines that don't apply to a benchmark (e.g. RS on
ACAS Xu — RS is classification-robustness only) get em-dashes for all
verdict cells instead of zero counts.

Required LaTeX packages: ``booktabs``, ``multirow``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    BENCHMARK_DISPLAY,
    EXP1_BENCHMARKS,
    EXP1_SOUND_VERIFIERS,
    METHOD_DISPLAY,
    add_common_args,
    bold,
    count_verdicts,
    fmt_int,
    fmt_pct,
    fmt_seconds,
    italic,
    latex_document_wrapper,
    mean_wall_clock,
    percent_solved,
    read_csv_no_header,
    read_csv_rows,
    write_table,
)

# Sound-verifier folder names (read-only, from VNN-COMP)
VNNCOMP_BENCH_DIRS = {
    "acasxu_2023":          "2025_acasxu_2023",
    "collins_rul_cnn_2022": "2025_collins_rul_cnn_2022",
    "dist_shift_2023":      "2025_dist_shift_2023",
    "linearizenn_2024":     "2025_linearizenn_2024",
    "tllverify_2023":       "2025_tllverifybench_2023",
    "malbeware":            "2025_malbeware",
    "metaroom_2023":        "2025_metaroom_2023",
}

# (group, method) tuples in display order. Sound verifiers come first
# (4 of them per the revised methodology), then probabilistic baselines,
# then ours.
METHOD_GROUPS: list[tuple[str, str]] = (
    [("sound", m) for m in EXP1_SOUND_VERIFIERS]
    + [
        ("prob",  "hashemi_clipping"),
        ("prob",  "rs"),
        ("prob",  "saver"),
        ("prob",  "probstar"),
        ("ours",  "ours"),
    ]
)

# Probabilistic baselines that simply don't apply to certain benchmarks.
# (e.g. RS / SaVer / Hashemi-clipping require classification-robustness
# specs; ACAS Xu / regression specs are not classification.)
# When *every* row in a baseline×benchmark CSV is NOT_APPLICABLE we
# render em-dashes instead of zero-counts. We auto-detect that below.

EM_DASH = "---"


def _load_method_counts_and_wall(
    csv_dir: Path, bench: str, method: str,
) -> tuple[dict[str, int], float, bool]:
    """Return (verdict counts, mean wall-clock, all_na).

    ``all_na`` is True when every row is NOT_APPLICABLE — that signals
    the baseline simply doesn't run on this benchmark and the table
    should print em-dashes for the verdict cells.
    """
    if method in EXP1_SOUND_VERIFIERS:
        # All four sound verifiers expected to use the same VNN-COMP CSV
        # layout: ``sound_verifiers/<2025_bench>/<method>/results.csv``,
        # with αβ-CROWN as the historical default at the bench level.
        method_subdir = "" if method == "alpha_beta_crown" else method
        sub = VNNCOMP_BENCH_DIRS.get(bench, "")
        candidate_paths = []
        if method_subdir:
            candidate_paths.append(csv_dir / "sound_verifiers" / sub / method_subdir / "results.csv")
        candidate_paths.append(csv_dir / "sound_verifiers" / sub / "results.csv")
        rows = []
        for p in candidate_paths:
            rows = read_csv_no_header(p)
            if rows:
                break
        synthetic_rows = []
        wall = []
        for r in rows:
            if len(r) < 6:
                continue
            synthetic_rows.append({"verdict": r[4]})
            try:
                wall.append(float(r[5]))
            except ValueError:
                pass
        if not synthetic_rows:
            # No data for this sound verifier on this benchmark: render
            # zero counts (consistent with current αβ-CROWN behaviour).
            return {}, 0.0, False
        counts = count_verdicts(synthetic_rows, "verdict")
        mean_s = sum(wall) / len(wall) if wall else 0.0
        return counts, mean_s, False

    if method == "ours":
        path = csv_dir / f"exp1_{bench}_ours.csv"
        rows = read_csv_rows(path)
        counts = count_verdicts(rows, "verdict")
        mean_s = mean_wall_clock(rows, "wall_s")
        all_na = bool(rows) and all(
            (r.get("verdict", "").strip().upper() == "NOT_APPLICABLE") for r in rows
        )
        return counts, mean_s, all_na

    path = csv_dir / f"exp1_{bench}_{method}.csv"
    rows = read_csv_rows(path)
    counts = count_verdicts(rows, "verdict")
    mean_s = mean_wall_clock(rows, "wall_s")
    all_na = bool(rows) and all(
        (r.get("verdict", "").strip().upper() == "NOT_APPLICABLE") for r in rows
    )
    return counts, mean_s, all_na


def _row_tex(group: str, method: str, counts: dict[str, int],
             wall_s: float, all_na: bool) -> str:
    name = METHOD_DISPLAY.get(method, method)
    if group == "sound":
        name = italic(name)
    elif group == "ours":
        name = bold(name)

    if all_na:
        # Tool incompatible with this benchmark: em-dash all verdict cells.
        cells = [name] + [EM_DASH] * 5 + [EM_DASH, EM_DASH]
        return " & ".join(cells) + r" \\"

    # Verdict counts — using simplified UNSAT/SAT plus indeterminate.
    # NOTE: column-meaning convention — UNSAT (correct/proved-safe),
    # SAT (correct/found-cex). False UNSAT and False SAT counts are
    # tracked at audit time; if the CSV doesn't carry them, we leave
    # those columns blank (printed as 0 fallback).
    correct_unsat = counts.get("UNSAT", 0)
    correct_sat = counts.get("SAT", 0)
    false_unsat = counts.get("FALSE_UNSAT", 0)
    false_sat = counts.get("FALSE_SAT", 0)
    indeterminate = (
        counts.get("UNKNOWN", 0)
        + counts.get("TIMEOUT", 0)
        + counts.get("ERROR", 0)
        + counts.get("NOT_APPLICABLE", 0)
        + counts.get("SKIPPED", 0)
    )
    pct = percent_solved(counts)
    cells = [
        name,
        fmt_int(correct_unsat),
        fmt_int(correct_sat),
        fmt_int(false_unsat),
        fmt_int(false_sat),
        fmt_int(indeterminate),
        fmt_pct(pct) + r"\%",
        fmt_seconds(wall_s),
    ]
    return " & ".join(cells) + r" \\"


def build_table(csv_dir: Path) -> str:
    body = []
    body.append(r"% Required LaTeX packages: booktabs, multirow.")
    body.append(r"\begin{table}[t]")
    body.append(r"\centering")
    body.append(r"\caption{Exp~1 verdict counts and wall-clock per method,"
                r" per VNN-COMP-derived benchmark. Italic rows are sound"
                r" verifiers (read-only from VNN-COMP); bold rows are ours."
                r" Indeterminate aggregates UNKNOWN, TIMEOUT, ERROR,"
                r" NOT\_APPLICABLE, SKIPPED. Em-dashes (---) mark"
                r" tool/benchmark combinations that are not applicable"
                r" (e.g. classification-only baselines on regression"
                r" benchmarks).}")
    body.append(r"\label{tab:exp1_verdict_matrix}")
    body.append(r"\small")
    body.append(r"\begin{tabular}{ll rrrrr rr}")
    body.append(r"\toprule")
    body.append(
        r"Benchmark & Method "
        r"& \shortstack{correct\\UNSAT} & \shortstack{correct\\SAT} "
        r"& \shortstack{false\\UNSAT} & \shortstack{false\\SAT} "
        r"& Indet. & \% solved & wall (s) \\"
    )
    body.append(r"\midrule")

    n_methods = len(METHOD_GROUPS)
    for bi, bench in enumerate(EXP1_BENCHMARKS):
        bench_disp = BENCHMARK_DISPLAY.get(bench, bench)
        bench_label = r"\multirow{" + str(n_methods) + r"}{*}{" + bench_disp + r"}"
        for mi, (group, method) in enumerate(METHOD_GROUPS):
            counts, wall, all_na = _load_method_counts_and_wall(
                csv_dir, bench, method
            )
            cell0 = bench_label if mi == 0 else ""
            row_body = _row_tex(group, method, counts, wall, all_na)
            body.append(cell0 + " & " + row_body)
        if bi < len(EXP1_BENCHMARKS) - 1:
            body.append(r"\midrule")

    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")
    body.append(r"\end{table}")
    table_str = "\n".join(body)
    return latex_document_wrapper(table_str, caption="", label="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="tab1_exp1_verdict_matrix.tex")
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "tab1_exp1_verdict_matrix.tex")
    table_tex = build_table(args.csv_dir)
    write_table(output, table_tex)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
