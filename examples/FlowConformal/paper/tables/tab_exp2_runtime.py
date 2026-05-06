"""Table — Exp 2 mean wall-clock per (method × benchmark).

Companion to Figure 3. Only αβ-CROWN among sound verifiers.

Required LaTeX packages: ``booktabs``, ``rotating``.
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
    fmt_seconds,
    italic,
    latex_document_wrapper,
    mean_wall_clock,
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


def _load_wall(csv_dir: Path, bench: str, method: str) -> tuple[float, bool]:
    if method == "alpha_beta_crown":
        rows = read_csv_rows(csv_dir / f"exp2_{bench}_alpha_beta_crown.csv")
        all_na = bool(rows) and all(
            (r.get("verdict", "").strip().upper() == "NOT_APPLICABLE") for r in rows
        )
        return mean_wall_clock(rows, "wall_s"), all_na
    if method == "ours":
        rows = read_csv_rows(csv_dir / f"exp2_{bench}_ours.csv")
        all_na = bool(rows) and all(
            (r.get("verdict", "").strip().upper() == "NOT_APPLICABLE") for r in rows
        )
        return mean_wall_clock(rows, "wall_s"), all_na
    rows = read_csv_rows(csv_dir / f"exp2_{bench}_{method}.csv")
    all_na = bool(rows) and all(
        (r.get("verdict", "").strip().upper() == "NOT_APPLICABLE") for r in rows
    )
    return mean_wall_clock(rows, "wall_s"), all_na


def build_table(csv_dir: Path) -> str:
    body = []
    body.append(r"% Required LaTeX packages: booktabs, rotating.")
    body.append(r"\begin{table}[t]")
    body.append(r"\centering")
    body.append(r"\caption{Exp~2 mean wall-clock per instance (seconds)"
                r" by method and benchmark. Italic = sound verifier;"
                r" bold = ours. Em-dashes (---) mark tool/benchmark"
                r" combinations that are not applicable.}")
    body.append(r"\label{tab:exp2_runtime}")
    body.append(r"\small")
    col_spec = "l" + "r" * len(EXP2_BENCHMARKS)
    body.append(r"\begin{tabular}{" + col_spec + r"}")
    body.append(r"\toprule")
    header_cells = ["Method"] + [
        r"\rotatebox{60}{" + BENCHMARK_DISPLAY.get(b, b) + r"}"
        for b in EXP2_BENCHMARKS
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
        for bench in EXP2_BENCHMARKS:
            wall, all_na = _load_wall(csv_dir, bench, method)
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
    add_common_args(parser, default_output="tab_exp2_runtime.tex")
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "tab_exp2_runtime.tex")
    table_tex = build_table(args.csv_dir)
    write_table(output, table_tex)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
