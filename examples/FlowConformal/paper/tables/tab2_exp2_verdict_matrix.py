"""Table 2 — Exp 2 verdict matrix + % solved.

Rows: αβ-CROWN re-run (only sound verifier in Exp 2 — NeuralSAT, PyRAT,
NNV, Rover are intentionally dropped per the 2026-04 paper revision),
4 probabilistic baselines, ours; per benchmark.

Columns: correct UNSAT, correct SAT, false UNSAT, false SAT,
indeterminate (TIMEOUT + ERROR + NOT_APPLICABLE + UNKNOWN + SKIPPED),
% solved, mean wall-clock per instance.

Required LaTeX packages: ``booktabs``, ``multirow``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    BENCHMARK_DISPLAY,
    EXP2_BENCHMARKS,
    EXP2_SOUND_VERIFIERS,
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
    read_csv_rows,
    write_table,
)


METHOD_GROUPS: list[tuple[str, str]] = (
    [("sound", m) for m in EXP2_SOUND_VERIFIERS]
    + [
        ("prob",  "hashemi_clipping"),
        ("prob",  "rs"),
        ("prob",  "saver"),
        ("prob",  "probstar"),
        ("ours",  "ours"),
    ]
)

EM_DASH = "---"


def _load_method_counts_and_wall(
    csv_dir: Path, bench: str, method: str,
) -> tuple[dict[str, int], float, bool]:
    if method == "alpha_beta_crown":
        path = csv_dir / f"exp2_{bench}_alpha_beta_crown.csv"
        rows = read_csv_rows(path)
        counts = count_verdicts(rows, "verdict")
        mean_s = mean_wall_clock(rows, "wall_s")
        all_na = bool(rows) and all(
            (r.get("verdict", "").strip().upper() == "NOT_APPLICABLE") for r in rows
        )
        return counts, mean_s, all_na
    if method == "ours":
        path = csv_dir / f"exp2_{bench}_ours.csv"
        rows = read_csv_rows(path)
        counts = count_verdicts(rows, "verdict")
        mean_s = mean_wall_clock(rows, "wall_s")
        all_na = bool(rows) and all(
            (r.get("verdict", "").strip().upper() == "NOT_APPLICABLE") for r in rows
        )
        return counts, mean_s, all_na
    path = csv_dir / f"exp2_{bench}_{method}.csv"
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
        cells = [name] + [EM_DASH] * 5 + [EM_DASH, EM_DASH]
        return " & ".join(cells) + r" \\"

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
    body.append(r"\caption{Exp~2 verdict counts and wall-clock per method on"
                r" the four large-network probabilistic-scale benchmarks."
                r" $\alpha,\!\beta$-CROWN is re-run with the same per-instance"
                r" budget as the probabilistic methods and is the only sound"
                r" verifier shown for Exp~2. Bold rows are ours. Indeterminate"
                r" aggregates UNKNOWN, TIMEOUT, ERROR, NOT\_APPLICABLE,"
                r" SKIPPED. Em-dashes (---) mark tool/benchmark combinations"
                r" that are not applicable.}")
    body.append(r"\label{tab:exp2_verdict_matrix}")
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
    for bi, bench in enumerate(EXP2_BENCHMARKS):
        bench_disp = BENCHMARK_DISPLAY.get(bench, bench)
        bench_label = r"\multirow{" + str(n_methods) + r"}{*}{" + bench_disp + r"}"
        for mi, (group, method) in enumerate(METHOD_GROUPS):
            counts, wall, all_na = _load_method_counts_and_wall(
                csv_dir, bench, method
            )
            cell0 = bench_label if mi == 0 else ""
            row_body = _row_tex(group, method, counts, wall, all_na)
            body.append(cell0 + " & " + row_body)
        if bi < len(EXP2_BENCHMARKS) - 1:
            body.append(r"\midrule")

    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")
    body.append(r"\end{table}")
    return latex_document_wrapper("\n".join(body), caption="", label="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="tab2_exp2_verdict_matrix.tex")
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "tab2_exp2_verdict_matrix.tex")
    table_tex = build_table(args.csv_dir)
    write_table(output, table_tex)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
