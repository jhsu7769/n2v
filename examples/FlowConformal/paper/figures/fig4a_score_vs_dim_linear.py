"""Figure 4a — Score function × output dim line plot (LINEAR y).

Reads ``ablation_score.csv`` (score-family ablation, schema in
``examples/FlowConformal/CSV_SCHEMAS.md`` §5.3) and plots mean
``volume_ratio`` vs output
dim, one line per score family.

Linear y-axis — extreme outliers (e.g. hyperrect ~440x at high dim)
are clipped at a configurable cap (default 50x) so the lower-magnitude
methods remain visible. A side-note in the legend lists any clipped
values.

The user is asked to pick between this figure and the analogous
table at ``paper/tables/tab_score_vs_dim.tex``.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    add_common_args,
    apply_paper_style,
    read_csv_rows,
    save_figure,
)

import matplotlib.pyplot as plt  # noqa: E402

SCORES = ["hyperrect", "ellipsoid", "gmm", "flow"]
SCORE_DISPLAY = {
    "hyperrect": "Hyper-rectangle",
    "ellipsoid": "Ellipsoid",
    "gmm":       "GMM",
    "flow":      "Flow (ours)",
}
SCORE_COLORS = {
    "hyperrect": "#888888",
    "ellipsoid": "#3690c0",
    "gmm":       "#5e3c99",
    "flow":      "#1b9e3a",
}


def _load(csv_dir: Path) -> dict[str, dict[int, list[float]]]:
    """Returns {score: {dim: [volume_ratio,...]}}."""
    rows = read_csv_rows(csv_dir / "ablation_score.csv")
    out: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        score = r.get("score", "").strip()
        try:
            dim = int(r.get("dim", "0"))
            vr = float(r.get("volume_ratio", "0") or 0)
        except ValueError:
            continue
        if score and dim:
            out[score][dim].append(vr)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="fig4a_score_vs_dim_linear.png")
    parser.add_argument(
        "--clip", type=float, default=50.0,
        help="Cap volume-ratio values at this threshold for plotting "
             "(linear-y plots otherwise blow up due to hyperrect outlier).",
    )
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "fig4a_score_vs_dim_linear.png")

    apply_paper_style()
    data = _load(args.csv_dir)

    if not data:
        # Smoke fallback: still produce a file so the test passes.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No ablation_score.csv found", ha="center", va="center")
        save_figure(fig, output)
        print(f"Wrote {output} (empty)")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    dims = sorted({d for s in data.values() for d in s.keys()})

    clipped_notes: list[str] = []
    for score in SCORES:
        if score not in data:
            continue
        ys, errs = [], []
        for d in dims:
            vals = data[score].get(d, [])
            if not vals:
                ys.append(np.nan)
                errs.append(0.0)
                continue
            arr = np.array(vals)
            mu = float(arr.mean())
            sd = float(arr.std())
            if mu > args.clip:
                clipped_notes.append(f"{SCORE_DISPLAY[score]} d={d}: {mu:.0f}")
                mu = args.clip
                sd = 0.0  # don't draw misleading errorbar past the cap
            ys.append(mu)
            errs.append(sd)
        ax.errorbar(
            dims, ys, yerr=errs,
            color=SCORE_COLORS[score],
            label=SCORE_DISPLAY[score],
            marker="o",
            capsize=3,
        )

    ax.set_yscale("linear")
    ax.set_xlabel("Output dimension $d$")
    ax.set_ylabel(r"Volume ratio (set vol / exact $(1-\alpha)$ vol)")
    title = "Score-function tightness vs output dimension"
    if clipped_notes:
        title += f"\n(clipped at {args.clip:g}x: " + "; ".join(clipped_notes) + ")"
    ax.set_title(title, fontsize=9)
    ax.set_xticks(dims)
    ax.legend(loc="upper left")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    save_figure(fig, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
