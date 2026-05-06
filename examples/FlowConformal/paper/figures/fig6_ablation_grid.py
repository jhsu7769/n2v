"""Figure 6 — 3-panel ablation grid.

Panel A — AMLS hyperparameters (ρ ∈ {0.05, 0.1, 0.2}; mcmc-steps
          ∈ {5, 10, 20, 40}). y = % solved on probe; one line per axis.
Panel B — Conformal-parameter sweep (4 mini-axes: α, m, ell-off, β2).
          y = % solved; one mini-line per axis.
Panel C — Flow-training heatmap (n_train × epochs).
          color = % solved on probe.

All inputs come from the per-row ``ablation_*.csv`` files in
``examples/FlowConformal/experiments/exp_ablation/outputs/`` (or the
fake-data sibling).
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

# Panel A: AMLS axes
RHOS = ["0.05", "0.1", "0.2"]
MCMCS = ["5", "10", "20", "40"]

# Panel B: conformal axes
CONFORMAL_AXES = {
    "alpha":  (["0.001", "0.01", "0.05", "0.1"], r"$\alpha$"),
    "m":      (["500", "2000", "8000"],          r"$m$ (calib size)"),
    "elloff": (["0", "1", "5"],                   r"$\ell_{\rm off}$"),
    "beta2":  (["0.001", "0.01", "0.1"],          r"$\beta_2$"),
}

# Panel C: flow-training grid
N_TRAINS = [1000, 2000, 5000, 10000, 20000, 50000]
EPOCHS = [500, 1000, 2000, 5000]


def _pct_solved(csv_dir: Path, fname: str) -> float:
    rows = read_csv_rows(csv_dir / fname)
    counts = count_verdicts(rows, "verdict")
    return percent_solved(counts)


def _draw_amls_panel(ax, csv_dir: Path) -> None:
    rho_vals = [_pct_solved(csv_dir, f"ablation_amls_hparam_rho{r}.csv") for r in RHOS]
    mcmc_vals = [_pct_solved(csv_dir, f"ablation_amls_hparam_mcmc{m}.csv") for m in MCMCS]

    # Plot two lines on a shared ax with two x-axes (using categorical x).
    ax2 = ax.twiny()
    x_rho = np.arange(len(RHOS))
    x_mcmc = np.arange(len(MCMCS))
    ax.plot(x_rho, rho_vals, marker="o", color="#1b9e3a", label=r"$\rho$ (rare-event prob.)")
    ax2.plot(x_mcmc, mcmc_vals, marker="s", color="#3690c0", label=r"MCMC steps")

    ax.set_xticks(x_rho)
    ax.set_xticklabels(RHOS)
    ax.set_xlabel(r"AMLS $\rho$")
    ax2.set_xticks(x_mcmc)
    ax2.set_xticklabels(MCMCS)
    ax2.set_xlabel("MCMC steps")
    ax.set_ylabel(r"\% solved on probe")
    ax.set_title("(a) AMLS hyperparameters")
    ax.set_ylim(0, 105)

    # Combine legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center", fontsize=8)


def _draw_conformal_panel(ax, csv_dir: Path) -> None:
    colors = ["#1b9e3a", "#3690c0", "#e6550d", "#5e3c99"]
    handles = []
    for ci, (axis, (vals, label)) in enumerate(CONFORMAL_AXES.items()):
        ys = [_pct_solved(csv_dir, f"ablation_conformal_params_{axis}{v}.csv") for v in vals]
        xs = np.arange(len(vals))
        h, = ax.plot(xs / max(1, len(vals) - 1), ys, marker="o",
                     color=colors[ci], label=label)
        handles.append(h)

    ax.set_xlabel("Sweep position (low → high)")
    ax.set_ylabel(r"\% solved on probe")
    ax.set_title("(b) Conformal parameters")
    ax.set_ylim(0, 105)
    ax.legend(handles=handles, loc="lower right", fontsize=8)


def _draw_flow_training_panel(ax, csv_dir: Path) -> None:
    grid = np.zeros((len(EPOCHS), len(N_TRAINS)))
    for ei, E in enumerate(EPOCHS):
        for ni, N in enumerate(N_TRAINS):
            grid[ei, ni] = _pct_solved(csv_dir, f"ablation_flow_training_n{N}_e{E}.csv")

    im = ax.imshow(grid, cmap="viridis", aspect="auto", origin="lower", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(N_TRAINS)))
    ax.set_xticklabels([f"{n//1000}K" for n in N_TRAINS])
    ax.set_yticks(np.arange(len(EPOCHS)))
    ax.set_yticklabels([str(e) for e in EPOCHS])
    ax.set_xlabel("Training samples $n$")
    ax.set_ylabel("Flow training epochs")
    ax.set_title("(c) Flow training")

    # Annotate
    for ei in range(len(EPOCHS)):
        for ni in range(len(N_TRAINS)):
            ax.text(ni, ei, f"{grid[ei, ni]:.0f}",
                    ha="center", va="center", color="white", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label=r"\% solved")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="fig6_ablation_grid.png")
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "fig6_ablation_grid.png")

    apply_paper_style()
    # Disable LaTeX text-rendering for tightening compat
    plt.rcParams["text.usetex"] = False

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    _draw_amls_panel(axes[0], args.csv_dir)
    _draw_conformal_panel(axes[1], args.csv_dir)
    _draw_flow_training_panel(axes[2], args.csv_dir)

    fig.tight_layout()
    save_figure(fig, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
