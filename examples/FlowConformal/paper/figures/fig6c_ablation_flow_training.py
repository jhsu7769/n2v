"""Figure 6c — Flow-training heatmap (single panel).

n_train (rows) × flow_epochs (cols); cell color = % solved on probe.
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

N_TRAINS = [1000, 2000, 5000, 10000, 20000, 50000]
EPOCHS = [500, 1000, 2000, 5000]


def _pct_solved(csv_dir: Path, fname: str) -> float:
    rows = read_csv_rows(csv_dir / fname)
    counts = count_verdicts(rows, "verdict")
    return percent_solved(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="fig6c_ablation_flow_training.png")
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "fig6c_ablation_flow_training.png")

    apply_paper_style()
    plt.rcParams["text.usetex"] = False

    fig, ax = plt.subplots(figsize=(7, 4))
    grid = np.zeros((len(EPOCHS), len(N_TRAINS)))
    for ei, E in enumerate(EPOCHS):
        for ni, N in enumerate(N_TRAINS):
            grid[ei, ni] = _pct_solved(
                args.csv_dir, f"ablation_flow_training_n{N}_e{E}.csv"
            )

    im = ax.imshow(grid, cmap="viridis", aspect="auto",
                   origin="lower", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(N_TRAINS)))
    ax.set_xticklabels([f"{n // 1000}K" for n in N_TRAINS])
    ax.set_yticks(np.arange(len(EPOCHS)))
    ax.set_yticklabels([str(e) for e in EPOCHS])
    ax.set_xlabel("Training samples $n$")
    ax.set_ylabel("Flow training epochs")
    ax.set_title("(c) Flow training")

    for ei in range(len(EPOCHS)):
        for ni in range(len(N_TRAINS)):
            ax.text(ni, ei, f"{grid[ei, ni]:.0f}",
                    ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label=r"% solved")

    fig.tight_layout()
    save_figure(fig, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
