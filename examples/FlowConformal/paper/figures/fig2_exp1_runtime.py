"""Figure 2 — Exp 1 runtime per benchmark per method.

Mean wall-clock per (method, benchmark), rendered as grouped bars with
explicit white-space between benchmark groups, hatched bars to
disambiguate methods, and a legend placed *outside* the plot area.

x-axis = benchmark; y-axis = wall-clock (s). Log-y is used only when
the dynamic range across the bars exceeds 50x; otherwise we stay
linear.

Methods plotted: ours + 4 probabilistic baselines. Sound verifiers are
relegated to Table 1 to keep the figure legible (they would otherwise
contribute very different magnitude bars).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    BENCHMARK_DISPLAY,
    EXP1_BENCHMARKS,
    METHOD_COLORS,
    METHOD_DISPLAY,
    add_common_args,
    apply_paper_style,
    read_csv_rows,
    save_figure,
)

import matplotlib.pyplot as plt  # noqa: E402

# Order matters for legend layout.
METHODS = ["hashemi_clipping", "rs", "saver", "probstar", "ours"]

# Hatch patterns to give each bar a tactile signature even in B&W.
METHOD_HATCH = {
    "hashemi_clipping": "//",
    "rs":               "..",
    "saver":            "xx",
    "probstar":         "\\\\",
    "ours":             "",
}


def _wall_stats(rows: list[dict[str, str]], wall_key: str) -> tuple[float, float]:
    vals = []
    for r in rows:
        s = r.get(wall_key, "").strip()
        if not s:
            continue
        try:
            vals.append(float(s))
        except ValueError:
            continue
    if not vals:
        return 0.0, 0.0
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std())


def _load(csv_dir: Path, bench: str, method: str) -> tuple[float, float]:
    if method == "ours":
        rows = read_csv_rows(csv_dir / f"exp1_{bench}_ours.csv")
        return _wall_stats(rows, "wall_s")
    rows = read_csv_rows(csv_dir / f"exp1_{bench}_{method}.csv")
    return _wall_stats(rows, "wall_s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="fig2_exp1_runtime.png")
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "fig2_exp1_runtime.png")

    apply_paper_style()

    n_bench = len(EXP1_BENCHMARKS)
    n_method = len(METHODS)
    # Wider gap between benchmark groups: cluster width 0.7 of unit, leaving
    # 0.3 of unit as gap.
    cluster_width = 0.7
    bar_width = cluster_width / n_method
    x_centers = np.arange(n_bench) * 1.0  # unit spacing per group

    means_grid = np.zeros((n_method, n_bench))
    stds_grid = np.zeros((n_method, n_bench))
    for mi, method in enumerate(METHODS):
        for bi, bench in enumerate(EXP1_BENCHMARKS):
            mu, sd = _load(args.csv_dir, bench, method)
            means_grid[mi, bi] = mu
            stds_grid[mi, bi] = sd

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for mi, method in enumerate(METHODS):
        offset = (mi - (n_method - 1) / 2) * bar_width
        ax.bar(
            x_centers + offset, means_grid[mi], bar_width,
            yerr=stds_grid[mi], capsize=2,
            color=METHOD_COLORS.get(method, "#888"),
            label=METHOD_DISPLAY.get(method, method),
            edgecolor="black", linewidth=0.5,
            hatch=METHOD_HATCH.get(method, ""),
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [BENCHMARK_DISPLAY.get(b, b) for b in EXP1_BENCHMARKS],
        rotation=20, ha="right",
    )
    ax.set_ylabel("Wall-clock per instance (s)")
    ax.set_title("Exp 1 runtime per benchmark per method")

    # Vertical separator lines between benchmark groups for clearer grouping.
    for i in range(1, n_bench):
        ax.axvline(x_centers[i] - 0.5, color="gray", linewidth=0.5,
                   alpha=0.3, linestyle=":")

    # Decide log vs linear y based on dynamic range across non-zero bars.
    nonzero = means_grid[means_grid > 0]
    if nonzero.size and nonzero.max() / max(nonzero.min(), 1e-9) > 50.0:
        ax.set_yscale("log")
    # else: keep linear

    # Legend OUTSIDE the plot, on the right.
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0, fontsize=9, frameon=False,
    )

    fig.tight_layout()
    save_figure(fig, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
