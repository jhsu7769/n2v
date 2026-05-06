"""Figure 5a — Wall-clock vs network parameter count, *linear y*.

Companion to ``fig5b_scaling_semilog.py`` (semi-log y). Both filter
out NOT_APPLICABLE rows so the line plots reflect only the runs that
actually completed.

x-axis is still log-scaled (parameter counts span ACAS Xu ~13K to ViT
~22M). The y-axis is plain linear. Use this to spot crossovers in
absolute wall-clock; the semi-log version is better for relative
comparisons across multiple decades.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    EXP1_BENCHMARKS,
    EXP2_BENCHMARKS,
    METHOD_COLORS,
    METHOD_DISPLAY,
    add_common_args,
    apply_paper_style,
    read_csv_no_header,
    read_csv_rows,
    save_figure,
)

import matplotlib.pyplot as plt  # noqa: E402

# Approximate parameter counts. TODO before final paper: count ONNX nodes precisely.
BENCHMARK_PARAM_COUNT = {
    "acasxu_2023":          13_500,
    "collins_rul_cnn_2022": 25_000,
    "dist_shift_2023":      50_000,
    "linearizenn_2024":     9_000,
    "tllverify_2023":       100_000,
    "malbeware":            150_000,
    "metaroom_2023":        500_000,
    "vit_2023":             22_000_000,
    "tinyimagenet_2024":    2_500_000,
    "cifar100_2024":        2_500_000,
    "cifar10_resnet110":    1_700_000,
}

VNNCOMP_BENCH_DIRS = {
    "acasxu_2023":          "2025_acasxu_2023",
    "collins_rul_cnn_2022": "2025_collins_rul_cnn_2022",
    "dist_shift_2023":      "2025_dist_shift_2023",
    "linearizenn_2024":     "2025_linearizenn_2024",
    "tllverify_2023":       "2025_tllverifybench_2023",
    "malbeware":            "2025_malbeware",
    "metaroom_2023":        "2025_metaroom_2023",
}

METHODS = ["alpha_beta_crown", "hashemi_clipping", "rs", "saver", "probstar", "ours"]


def _wall_mean_skip_na(rows: list[dict[str, str]], wall_key: str) -> float:
    """Mean wall-clock, ignoring NOT_APPLICABLE rows."""
    vals = []
    for r in rows:
        verdict = r.get("verdict", "").strip().upper()
        if verdict == "NOT_APPLICABLE":
            continue
        s = r.get(wall_key, "").strip()
        if not s:
            continue
        try:
            vals.append(float(s))
        except ValueError:
            continue
    return float(np.mean(vals)) if vals else 0.0


def _vnncomp_wall_mean(csv_dir: Path, bench: str) -> float:
    rel = VNNCOMP_BENCH_DIRS.get(bench)
    if rel is None:
        return 0.0
    rows = read_csv_no_header(csv_dir / "sound_verifiers" / rel / "results.csv")
    vals = []
    for r in rows:
        if len(r) < 6:
            continue
        verdict = r[4].strip().upper()
        if verdict == "NOT_APPLICABLE":
            continue
        try:
            vals.append(float(r[5]))
        except ValueError:
            continue
    return float(np.mean(vals)) if vals else 0.0


def _exp1_load(csv_dir: Path, bench: str, method: str) -> float:
    if method == "ours":
        return _wall_mean_skip_na(
            read_csv_rows(csv_dir / f"exp1_{bench}_ours.csv"), "wall_s"
        )
    if method == "alpha_beta_crown":
        return _vnncomp_wall_mean(csv_dir, bench)
    return _wall_mean_skip_na(
        read_csv_rows(csv_dir / f"exp1_{bench}_{method}.csv"), "wall_s"
    )


def _exp2_load(csv_dir: Path, bench: str, method: str) -> float:
    if method == "ours":
        return _wall_mean_skip_na(
            read_csv_rows(csv_dir / f"exp2_{bench}_ours.csv"), "wall_s"
        )
    if method == "alpha_beta_crown":
        return _wall_mean_skip_na(
            read_csv_rows(csv_dir / f"exp2_{bench}_alpha_beta_crown.csv"), "wall_s"
        )
    return _wall_mean_skip_na(
        read_csv_rows(csv_dir / f"exp2_{bench}_{method}.csv"), "wall_s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="fig5a_scaling_linear.png")
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "fig5a_scaling_linear.png")

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method in METHODS:
        xs, ys = [], []
        for bench in EXP1_BENCHMARKS:
            wall = _exp1_load(args.csv_dir, bench, method)
            if wall > 0:
                xs.append(BENCHMARK_PARAM_COUNT[bench])
                ys.append(wall)
        for bench in EXP2_BENCHMARKS:
            wall = _exp2_load(args.csv_dir, bench, method)
            if wall > 0:
                xs.append(BENCHMARK_PARAM_COUNT[bench])
                ys.append(wall)

        if not xs:
            continue
        order = np.argsort(xs)
        xs_arr = np.array(xs)[order]
        ys_arr = np.array(ys)[order]
        ax.plot(
            xs_arr, ys_arr,
            color=METHOD_COLORS.get(method, "#888"),
            marker="o",
            label=METHOD_DISPLAY.get(method, method),
        )

    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.set_xlabel("Network parameter count (log scale)")
    ax.set_ylabel("Wall-clock per instance (s)")
    ax.set_title("Scaling: wall-clock vs network size (linear y)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    save_figure(fig, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
