"""Figure 6b — Conformal-parameter ablation (single panel).

4 mini-axes (alpha, m, ell-off, beta2) collapsed onto a normalised
sweep position (low → high). y = % solved on probe.
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

CONFORMAL_AXES = {
    "alpha":  (["0.001", "0.01", "0.05", "0.1"], r"$\alpha$"),
    "m":      (["500", "2000", "8000"],          r"$m$ (calibration size)"),
    "elloff": (["0", "1", "5"],                   r"$\ell_{\rm off}$"),
    "beta2":  (["0.001", "0.01", "0.1"],          r"$\beta_2$"),
}


def _pct_solved(csv_dir: Path, fname: str) -> float:
    rows = read_csv_rows(csv_dir / fname)
    counts = count_verdicts(rows, "verdict")
    return percent_solved(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="fig6b_ablation_conformal_params.png")
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "fig6b_ablation_conformal_params.png")

    apply_paper_style()
    plt.rcParams["text.usetex"] = False

    colors = ["#1b9e3a", "#3690c0", "#e6550d", "#5e3c99"]
    markers = ["o", "s", "D", "^"]
    fig, ax = plt.subplots(figsize=(6, 4))

    handles = []
    for ci, (axis, (vals, label)) in enumerate(CONFORMAL_AXES.items()):
        ys = [_pct_solved(args.csv_dir, f"ablation_conformal_params_{axis}{v}.csv")
              for v in vals]
        xs = np.linspace(0.0, 1.0, len(vals))
        h, = ax.plot(xs, ys, marker=markers[ci],
                     color=colors[ci], label=label)
        handles.append(h)

    ax.set_xlabel("Sweep position (low to high)")
    ax.set_ylabel(r"% solved on probe")
    ax.set_title("(b) Conformal parameters")
    ax.set_ylim(0, 105)
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    save_figure(fig, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
