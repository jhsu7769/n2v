"""Table — score-family rows × dim columns; cells = mean volume ratio.

Companion table to Figure 4a; the user picks between figure and table.

Reads ``ablation_score.csv`` (schema in
``examples/FlowConformal/CSV_SCHEMAS.md`` §5.3).
Columns are output dimensions (sorted ascending); rows are score
families (hyperrect / ellipsoid / GMM / flow). Each cell is the mean
volume ratio across seeds, formatted as ``{mean:.2f}`` for ratios <100,
``{mean:.0f}`` for ratios >=100.

Required LaTeX packages: ``booktabs``.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    add_common_args,
    bold,
    latex_document_wrapper,
    read_csv_rows,
    write_table,
)


SCORES = ["hyperrect", "ellipsoid", "gmm", "flow"]
SCORE_DISPLAY = {
    "hyperrect": "Hyper-rectangle",
    "ellipsoid": "Ellipsoid",
    "gmm":       "GMM",
    "flow":      "Flow (ours)",
}


def _load(csv_dir: Path) -> dict[str, dict[int, list[float]]]:
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


def _fmt(v: float) -> str:
    if v != v:  # NaN
        return "--"
    if v >= 100.0:
        return f"{v:.0f}"
    return f"{v:.2f}"


def build_table(csv_dir: Path) -> str:
    data = _load(csv_dir)
    dims = sorted({d for s in data.values() for d in s.keys()})

    body = []
    body.append(r"% Required LaTeX packages: booktabs.")
    body.append(r"\begin{table}[t]")
    body.append(r"\centering")
    body.append(r"\caption{Score-family tightness vs output dimension."
                r" Each cell is the mean volume ratio (set volume divided"
                r" by exact $(1-\alpha)$ reach-set volume) across seeds."
                r" Lower is tighter; 1.0 is the floor. Bold row is ours.}")
    body.append(r"\label{tab:score_vs_dim}")
    body.append(r"\small")
    col_spec = "l" + "r" * len(dims)
    body.append(r"\begin{tabular}{" + col_spec + r"}")
    body.append(r"\toprule")
    header = ["Score family"] + [f"$d={d}$" for d in dims]
    body.append(" & ".join(header) + r" \\")
    body.append(r"\midrule")

    for score in SCORES:
        if score not in data and not dims:
            continue
        cells = []
        name = SCORE_DISPLAY[score]
        if score == "flow":
            name = bold(name)
        cells.append(name)
        for d in dims:
            vals = data.get(score, {}).get(d, [])
            if not vals:
                cells.append("--")
            else:
                mu = sum(vals) / len(vals)
                cells.append(_fmt(mu))
        body.append(" & ".join(cells) + r" \\")

    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")
    body.append(r"\end{table}")
    return latex_document_wrapper("\n".join(body), caption="", label="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="tab_score_vs_dim.tex")
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "tab_score_vs_dim.tex")
    table_tex = build_table(args.csv_dir)
    write_table(output, table_tex)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
