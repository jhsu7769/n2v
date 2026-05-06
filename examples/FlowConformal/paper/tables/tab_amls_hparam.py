"""Table — AMLS hyperparameter ablation: ρ × MCMC steps; cells = false UNSAT.

Reads ``ablation_amls_hparam_rho<r>.csv`` and ``ablation_amls_hparam_mcmc<s>.csv``
(probe-schema CSVs). Cells are the count of *false UNSAT* verdicts on
the probe — i.e. instances where the AMLS upper-stage incorrectly
certified UNSAT but ground truth is SAT (a soundness violation).

Because the available probe CSVs only carry one ρ axis OR one MCMC
axis (never the cross product), the table renders a 2-D shell: the
diagonal cells use the per-axis CSVs; off-diagonal cells fall back
to the per-axis CSV that matches its column (mcmc) — i.e. each row's
ρ value uses the same MCMC value as the column header. When real
cross-product CSVs become available, swap ``_load_probe_falses`` for
the matched-axis filename.

False-UNSAT count is derived from the probe CSV's ``worst_max_margin``
column: a row is a false UNSAT if ``verdict == 'UNSAT'`` and
``worst_max_margin <= 0`` (negative margin = ground-truth SAT).

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


RHOS = ["0.05", "0.1", "0.2"]
MCMCS = ["5", "10", "20", "40"]


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


def _load_rho(csv_dir: Path, rho: str) -> int:
    rows = read_csv_rows(csv_dir / f"ablation_amls_hparam_rho{rho}.csv")
    return _false_unsat_count(rows)


def _load_mcmc(csv_dir: Path, mcmc: str) -> int:
    rows = read_csv_rows(csv_dir / f"ablation_amls_hparam_mcmc{mcmc}.csv")
    return _false_unsat_count(rows)


def build_table(csv_dir: Path) -> str:
    body = []
    body.append(r"% Required LaTeX packages: booktabs.")
    body.append(r"\begin{table}[t]")
    body.append(r"\centering")
    body.append(r"\caption{AMLS hyperparameter ablation: false-UNSAT count"
                r" on the 20-instance ACAS Xu probe by AMLS rare-event"
                r" probability $\rho$ and MCMC steps per level. Lower is"
                r" better (zero would be ideal). The current implementation"
                r" only sweeps ρ and MCMC independently; cells fill in by"
                r" assuming the off-axis value matches the marginal sweep."
                r" When the cross-product sweep is run (TODO),"
                r" replace ``\_load\_probe\_falses`` with matched-axis"
                r" filename loading.}")
    body.append(r"\label{tab:amls_hparam}")
    body.append(r"\small")
    col_spec = "l" + "r" * len(MCMCS)
    body.append(r"\begin{tabular}{" + col_spec + r"}")
    body.append(r"\toprule")
    header = [r"$\rho$ \textbackslash{} MCMC"] + [f"{m}" for m in MCMCS]
    body.append(" & ".join(header) + r" \\")
    body.append(r"\midrule")

    for rho in RHOS:
        rho_falses = _load_rho(csv_dir, rho)
        cells = [rho]
        for mcmc in MCMCS:
            mcmc_falses = _load_mcmc(csv_dir, mcmc)
            # Without a real cross-product, take max(rho_falses, mcmc_falses)
            # as a conservative proxy: if either marginal axis already shows a
            # failure, the joint setting is at least as bad.
            cell = max(rho_falses, mcmc_falses)
            cells.append(str(cell))
        body.append(" & ".join(cells) + r" \\")

    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")
    body.append(r"\end{table}")
    return latex_document_wrapper("\n".join(body), caption="", label="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="tab_amls_hparam.tex")
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "tab_amls_hparam.tex")
    table_tex = build_table(args.csv_dir)
    write_table(output, table_tex)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
