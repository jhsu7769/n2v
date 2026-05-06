"""Table — Exp 1 mean wall-clock per (method × benchmark).

Companion to Figure 2; the user can pick figure or table.

Rows = methods (αβ-CROWN, NeuralSAT, PyRAT, CORA, hashemi_clipping,
RS, SaVer, ProbStar, ours), cols = benchmarks. Cells = mean wall-clock
per instance (s). Em-dash for not-applicable combinations.

Required LaTeX packages: ``booktabs``, ``rotating`` (for sideways
column headers).
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
    fmt_seconds,
    italic,
    latex_document_wrapper,
    mean_wall_clock,
    read_csv_no_header,
    read_csv_rows,
    write_table,
)


VNNCOMP_BENCH_DIRS = {
    "acasxu_2023":          "2025_acasxu_2023",
    "collins_rul_cnn_2022": "2025_collins_rul_cnn_2022",
    "dist_shift_2023":      "2025_dist_shift_2023",
    "linearizenn_2024":     "2025_linearizenn_2024",
    "tllverify_2023":       "2025_tllverifybench_2023",
    "malbeware":            "2025_malbeware",
    "metaroom_2023":        "2025_metaroom_2023",
}

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

EM_DASH = "---"


def _sound_wall(csv_dir: Path, bench: str, method: str) -> tuple[float, bool]:
    """Returns (mean_wall, all_na)."""
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
    if not rows:
        return 0.0, False
    walls = []
    for r in rows:
        if len(r) < 6:
            continue
        try:
            walls.append(float(r[5]))
        except ValueError:
            continue
    return (sum(walls) / len(walls)) if walls else 0.0, False


def _baseline_wall(csv_dir: Path, bench: str, method: str) -> tuple[float, bool]:
    rows = read_csv_rows(csv_dir / f"exp1_{bench}_{method}.csv")
    all_na = bool(rows) and all(
        (r.get("verdict", "").strip().upper() == "NOT_APPLICABLE") for r in rows
    )
    return mean_wall_clock(rows, "wall_s"), all_na


def _ours_wall(csv_dir: Path, bench: str) -> tuple[float, bool]:
    rows = read_csv_rows(csv_dir / f"exp1_{bench}_ours.csv")
    all_na = bool(rows) and all(
        (r.get("verdict", "").strip().upper() == "NOT_APPLICABLE") for r in rows
    )
    return mean_wall_clock(rows, "wall_s"), all_na


def build_table(csv_dir: Path) -> str:
    body = []
    body.append(r"% Required LaTeX packages: booktabs, rotating.")
    body.append(r"\begin{table}[t]")
    body.append(r"\centering")
    body.append(r"\caption{Exp~1 mean wall-clock per instance (seconds)"
                r" by method and benchmark. Italic rows are sound verifiers"
                r" (read-only from VNN-COMP); the bold row is ours."
                r" Em-dashes (---) mark tool/benchmark combinations that"
                r" are not applicable.}")
    body.append(r"\label{tab:exp1_runtime}")
    body.append(r"\small")
    col_spec = "l" + "r" * len(EXP1_BENCHMARKS)
    body.append(r"\begin{tabular}{" + col_spec + r"}")
    body.append(r"\toprule")
    header_cells = ["Method"] + [
        r"\rotatebox{60}{" + BENCHMARK_DISPLAY.get(b, b) + r"}"
        for b in EXP1_BENCHMARKS
    ]
    body.append(" & ".join(header_cells) + r" \\")
    body.append(r"\midrule")

    last_group = None
    for group, method in METHOD_GROUPS:
        if last_group is not None and group != last_group:
            body.append(r"\midrule")
        last_group = group
        name = METHOD_DISPLAY.get(method, method)
        if group == "sound":
            name = italic(name)
        elif group == "ours":
            name = bold(name)
        cells = [name]
        for bench in EXP1_BENCHMARKS:
            if method in EXP1_SOUND_VERIFIERS:
                wall, all_na = _sound_wall(csv_dir, bench, method)
            elif method == "ours":
                wall, all_na = _ours_wall(csv_dir, bench)
            else:
                wall, all_na = _baseline_wall(csv_dir, bench, method)
            if all_na:
                cells.append(EM_DASH)
            else:
                cells.append(fmt_seconds(wall))
        body.append(" & ".join(cells) + r" \\")

    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")
    body.append(r"\end{table}")
    return latex_document_wrapper("\n".join(body), caption="", label="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="tab_exp1_runtime.tex")
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "tab_exp1_runtime.tex")
    table_tex = build_table(args.csv_dir)
    write_table(output, table_tex)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
