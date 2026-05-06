"""Table — flow-training ablation: n_train rows × flow_epochs cols.

Cells = false-UNSAT count on the 20-instance ACAS Xu probe (lower is
better). Reads ``ablation_flow_training_n<N>_e<E>.csv`` from the
same probe schema.

Required LaTeX packages: ``booktabs``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    add_common_args,
    latex_document_wrapper,
    read_csv_rows,
    write_table,
)


N_TRAINS = [1000, 2000, 5000, 10000, 20000, 50000]
EPOCHS = [500, 1000, 2000, 5000]


def _false_unsat_count(rows: list[dict[str, str]]) -> int:
    n = 0
    for r in rows:
        v = r.get("verdict", "").strip().upper()
        if v != "UNSAT":
            continue
        try:
            wmm = float(r.get("worst_max_margin", "") or 0.0)
        except ValueError:
            continue
        if wmm <= 0.0:
            n += 1
    return n


def build_table(csv_dir: Path) -> str:
    body = []
    body.append(r"% Required LaTeX packages: booktabs.")
    body.append(r"\begin{table}[t]")
    body.append(r"\centering")
    body.append(r"\caption{Flow-training ablation: false-UNSAT count on the"
                r" 20-instance ACAS Xu probe by training-set size $n$ (rows)"
                r" and flow epochs $E$ (columns). Lower is better; zero is"
                r" the soundness ideal.}")
    body.append(r"\label{tab:flow_training}")
    body.append(r"\small")
    col_spec = "l" + "r" * len(EPOCHS)
    body.append(r"\begin{tabular}{" + col_spec + r"}")
    body.append(r"\toprule")
    header = ["$n$ \\textbackslash{} $E$"] + [f"{e}" for e in EPOCHS]
    body.append(" & ".join(header) + r" \\")
    body.append(r"\midrule")

    for N in N_TRAINS:
        cells = [f"{N // 1000}K"]
        for E in EPOCHS:
            rows = read_csv_rows(csv_dir / f"ablation_flow_training_n{N}_e{E}.csv")
            cells.append(str(_false_unsat_count(rows)))
        body.append(" & ".join(cells) + r" \\")

    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")
    body.append(r"\end{table}")
    return latex_document_wrapper("\n".join(body), caption="", label="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="tab_flow_training.tex")
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "tab_flow_training.tex")
    table_tex = build_table(args.csv_dir)
    write_table(output, table_tex)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
