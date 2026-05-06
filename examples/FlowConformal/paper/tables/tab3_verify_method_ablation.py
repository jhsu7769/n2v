"""Table 3 — Verification-method ablation.

Five rows (scenario, scenario_v2, AMLS, IS-tilted, Langevin) × verdict
counts + mean wall-clock. Reads the
`ablation_verify_method_<method>.csv` files written by
`ablation_run_verify_method.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    add_common_args,
    bold,
    count_verdicts,
    fmt_int,
    fmt_pct,
    fmt_seconds,
    latex_document_wrapper,
    mean_wall_clock,
    percent_solved,
    read_csv_rows,
    write_table,
)


METHODS = ["scenario", "scenario_v2", "amls", "is_tilted", "langevin"]
METHOD_LABELS = {
    "scenario":    "Scenario (orig.)",
    "scenario_v2": "Scenario v2",
    "amls":        "AMLS",
    "is_tilted":   "IS (tilted)",
    "langevin":    "Langevin",
}
DEFAULT_METHOD = "amls"  # bolded as the chosen default


def _load(csv_dir: Path, method: str) -> tuple[dict[str, int], float]:
    path = csv_dir / f"ablation_verify_method_{method}.csv"
    rows = read_csv_rows(path)
    counts = count_verdicts(rows, "verdict")
    mean_s = mean_wall_clock(rows, "wall_s")
    return counts, mean_s


def _row_tex(method: str, counts: dict[str, int], wall_s: float) -> str:
    name = METHOD_LABELS[method]
    if method == DEFAULT_METHOD:
        name = bold(name)
    pct = percent_solved(counts)
    cells = [
        name,
        fmt_int(counts.get("UNSAT", 0)),
        fmt_int(counts.get("SAT", 0)),
        fmt_int(counts.get("UNKNOWN", 0)),
        fmt_int(counts.get("ERROR", 0) + counts.get("TIMEOUT", 0)),
        fmt_pct(pct) + r"\%",
        fmt_seconds(wall_s),
    ]
    return " & ".join(cells) + r" \\"


def build_table(csv_dir: Path) -> str:
    body = []
    body.append(r"\begin{table}[t]")
    body.append(r"\centering")
    body.append(r"\caption{Verification-method ablation (Exp~ablation Phase~A)."
                r" Each row replaces the upper-stage verifier on the same"
                r" 20-instance ACAS Xu probe; flow training and conformal"
                r" calibration are held fixed. Bold = our default.}")
    body.append(r"\label{tab:verify_method_ablation}")
    body.append(r"\small")
    body.append(r"\begin{tabular}{l rrrr rr}")
    body.append(r"\toprule")
    body.append(
        r"Method & UNSAT & SAT & UNK & T/E & \% solved & wall (s) \\"
    )
    body.append(r"\midrule")

    for method in METHODS:
        counts, wall = _load(csv_dir, method)
        body.append(_row_tex(method, counts, wall))

    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")
    body.append(r"\end{table}")
    return latex_document_wrapper("\n".join(body), caption="", label="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="tab3_verify_method_ablation.tex")
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "tab3_verify_method_ablation.tex")
    table_tex = build_table(args.csv_dir)
    write_table(output, table_tex)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
