"""Exp 4 scaling table — depth × method matrix of mean wall-clock + TIMEOUT counts.

Rows: synthetic-network depths (2, 4, 8, 16, 24, 32, 40, width=512).
Columns: method (αβ-CROWN, NeuralSAT, Hashemi-clip, Ours).
Each cell: mean wall-clock (s) over the 10 instances at that depth,
with a "(N/M TO)" annotation when ``N`` of ``M`` instances hit the
per-instance shell timeout.

Reads ``exp4_d<D>_<method>.csv`` from ``--csv-dir`` (REQUIRED — no
fake-data fallback). Writes a single ``.tex`` table file consumable
by the paper.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
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

EXP4_DEPTHS = (2, 4, 8, 16, 24, 32, 40)
EXP4_METHODS = ("alpha_beta_crown", "neuralsat", "hashemi_clipping", "ours")
EXP4_SOUND = {"alpha_beta_crown", "neuralsat"}


def _depth_method_cell(rows: list[dict[str, str]]) -> tuple[float, int, int]:
    """Return (mean_wall_s_over_solved, n_total, n_timeout)."""
    n_total = len(rows)
    n_timeout = sum(
        1 for r in rows
        if r.get("verdict", "").strip().upper() == "TIMEOUT"
    )
    solved_rows = [
        r for r in rows
        if r.get("verdict", "").strip().upper() not in {"TIMEOUT", "ERROR"}
    ]
    return mean_wall_clock(solved_rows, "wall_s"), n_total, n_timeout


def _format_cell(mean_wall: float, n_total: int, n_timeout: int) -> str:
    if n_total == 0:
        return "--"
    if n_timeout == n_total:
        return r"\textsc{to}"
    base = fmt_seconds(mean_wall)
    if n_timeout > 0:
        return rf"{base} \scriptsize{{({n_timeout}/{n_total}\,TO)}}"
    return base


def build_table(csv_dir: Path) -> str:
    body: list[str] = []
    body.append(r"% Required LaTeX packages: booktabs.")
    body.append(r"\begin{table}[t]")
    body.append(r"\centering")
    body.append(
        r"\caption{Exp~4 scaling on the synthetic 1-Lipschitz family"
        r" (width=512). Each cell shows mean wall-clock (s, over solved"
        r" instances) at the given depth; ``$(N/M$\,TO$)$'' marks how"
        r" many of $M{=}10$ instances hit the per-instance shell"
        r" timeout. Italic = sound verifier; bold = ours.}"
    )
    body.append(r"\label{tab:exp4_scaling}")
    body.append(r"\small")

    col_spec = "l" + "r" * len(EXP4_DEPTHS)
    body.append(rf"\begin{{tabular}}{{{col_spec}}}")
    body.append(r"\toprule")

    header_cells = ["Method"] + [
        rf"$d{{=}}{d}$" for d in EXP4_DEPTHS
    ]
    body.append(" & ".join(header_cells) + r" \\")
    body.append(r"\midrule")

    for method in EXP4_METHODS:
        name = METHOD_DISPLAY.get(method, method)
        if method in EXP4_SOUND:
            name = italic(name)
        elif method == "ours":
            name = bold(name)

        cells = [name]
        for depth in EXP4_DEPTHS:
            rows = read_csv_rows(csv_dir / f"exp4_d{depth}_{method}.csv")
            mean_wall, n_total, n_timeout = _depth_method_cell(rows)
            cells.append(_format_cell(mean_wall, n_total, n_timeout))
        body.append(" & ".join(cells) + r" \\")

    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")
    body.append(r"\end{table}")
    return "\n".join(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="tab_exp4_scaling.tex")
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "tab_exp4_scaling.tex")

    table_tex = latex_document_wrapper(
        build_table(args.csv_dir),
        caption="Exp~4 scaling table",
        label="tab:exp4_scaling",
    )
    write_table(output, table_tex)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
