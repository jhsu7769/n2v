"""Table — conformal-parameter ablation: one row per (knob, value).

Reads ``ablation_conformal_params_<axis><value>.csv`` for the 4 axes:
α, m, ell-off, β2. Each row reports false-UNSAT count, % verifiability
(i.e. % solved on probe), and the joint $(\\epsilon, \\delta)$
certificate where available (here approximated as $1 - \\%solved/100$).

Required LaTeX packages: ``booktabs``, ``multirow``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    add_common_args,
    count_verdicts,
    latex_document_wrapper,
    percent_solved,
    read_csv_rows,
    write_table,
)


CONFORMAL_AXES = {
    "alpha":  (["0.001", "0.01", "0.05", "0.1"], r"$\alpha$"),
    "m":      (["500", "2000", "8000"],          r"$m$"),
    "elloff": (["0", "1", "5"],                   r"$\ell_{\rm off}$"),
    "beta2":  (["0.001", "0.01", "0.1"],          r"$\beta_2$"),
}


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


def _verifiability_pct(rows: list[dict[str, str]]) -> float:
    counts = count_verdicts(rows, "verdict")
    return percent_solved(counts)


def build_table(csv_dir: Path) -> str:
    body = []
    body.append(r"% Required LaTeX packages: booktabs, multirow.")
    body.append(r"\begin{table}[t]")
    body.append(r"\centering")
    body.append(r"\caption{Conformal-parameter ablation. Each row varies"
                r" one knob; cells report false-UNSAT count on the probe,"
                r" \% verifiability (i.e. \% of instances solved), and"
                r" joint $\epsilon$ (approximated as $1 - $ verifiability"
                r" / 100; tighten when real epsilon-delta logging is"
                r" wired through).}")
    body.append(r"\label{tab:conformal_params}")
    body.append(r"\small")
    body.append(r"\begin{tabular}{ll rrr}")
    body.append(r"\toprule")
    body.append(r"Knob & Value & False UNSAT & \% verifiability & joint $\epsilon$ \\")
    body.append(r"\midrule")

    n_axes = len(CONFORMAL_AXES)
    for ai, (axis, (vals, label)) in enumerate(CONFORMAL_AXES.items()):
        knob_label = r"\multirow{" + str(len(vals)) + r"}{*}{" + label + r"}"
        for vi, v in enumerate(vals):
            rows = read_csv_rows(csv_dir / f"ablation_conformal_params_{axis}{v}.csv")
            f_unsat = _false_unsat_count(rows)
            pct = _verifiability_pct(rows)
            joint_eps = 1.0 - pct / 100.0
            cell0 = knob_label if vi == 0 else ""
            row_cells = [
                cell0,
                v,
                str(f_unsat),
                f"{pct:.1f}\\%",
                f"{joint_eps:.3f}",
            ]
            body.append(" & ".join(row_cells) + r" \\")
        if ai < n_axes - 1:
            body.append(r"\midrule")

    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")
    body.append(r"\end{table}")
    return latex_document_wrapper("\n".join(body), caption="", label="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="tab_conformal_params.tex")
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "tab_conformal_params.tex")
    table_tex = build_table(args.csv_dir)
    write_table(output, table_tex)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
