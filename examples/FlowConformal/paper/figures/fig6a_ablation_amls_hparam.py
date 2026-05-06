"""Figure 6a — AMLS hyperparameter ablation (single panel).

Reads ``ablation_amls_hparam_rho<r>.csv`` and
``ablation_amls_hparam_mcmc<s>.csv`` (probe-schema CSVs) and plots
% solved on the probe as a function of:
  - ρ (rare-event probability per AMLS level)
  - MCMC steps per level

Two lines on a shared axes with a secondary x-axis (one tick set per
sweep).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    add_common_args,
    apply_paper_style,
    count_verdicts,
    percent_solved,
    read_csv_rows,
    save_figure,
)

import matplotlib.pyplot as plt  # noqa: E402

RHOS = ["0.05", "0.1", "0.2"]
MCMCS = ["5", "10", "20", "40"]


def _pct_solved(csv_dir: Path, fname: str) -> float:
    rows = read_csv_rows(csv_dir / fname)
    counts = count_verdicts(rows, "verdict")
    return percent_solved(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="fig6a_ablation_amls_hparam.png")
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "fig6a_ablation_amls_hparam.png")

    apply_paper_style()
    plt.rcParams["text.usetex"] = False

    fig, ax = plt.subplots(figsize=(6, 4))
    rho_vals = [_pct_solved(args.csv_dir, f"ablation_amls_hparam_rho{r}.csv") for r in RHOS]
    mcmc_vals = [_pct_solved(args.csv_dir, f"ablation_amls_hparam_mcmc{m}.csv") for m in MCMCS]

    ax2 = ax.twiny()
    x_rho = np.arange(len(RHOS))
    x_mcmc = np.arange(len(MCMCS))

    ax.plot(x_rho, rho_vals, marker="o", color="#1b9e3a",
            label=r"$\rho$ (rare-event probability)")
    ax2.plot(x_mcmc, mcmc_vals, marker="s", color="#3690c0",
             linestyle="--", label="MCMC steps per level")

    ax.set_xticks(x_rho)
    ax.set_xticklabels(RHOS)
    ax.set_xlabel(r"AMLS $\rho$")
    ax2.set_xticks(x_mcmc)
    ax2.set_xticklabels(MCMCS)
    ax2.set_xlabel("MCMC steps")
    ax.set_ylabel(r"% solved on probe")
    ax.set_title("(a) AMLS hyperparameters")
    ax.set_ylim(0, 105)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center", fontsize=9)

    fig.tight_layout()
    save_figure(fig, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
