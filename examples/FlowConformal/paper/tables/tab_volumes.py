"""Comprehensive volume table — aggregates volume measurements across
Exp 3 benchmarks (3d banana, synthetic 1-Lip identity, geo-transforms)
and the score ablation, producing a single LaTeX table.

Source CSVs (all read from ``--csv-dir``, REQUIRED — no fake-data fallback):

  * ``exp3_3d_banana_ours.csv``
        ``volume_estimate``, ``volume_ratio_vs_exact``, ``coverage_empirical``
        (single-network / single-method = flow). Exact volume reconstructed
        via ``volume_estimate / volume_ratio_vs_exact``.

  * ``exp3_synthetic_ours.csv``
        Same schema with a ``dim`` column (5/10/20). Each row is one
        flow-conformal verification on the d-D 1-Lipschitz identity-act net.

  * ``exp3_geo_transforms_ours.csv``
        Per-network rows for axis-aligned and rotated identity nets.

  * ``ablation_score.csv``
        Per (network, score) rows giving the volume of every score family
        on Exp 3 benchmarks (hyperrect / ellipsoid / GMM / flow).

Output schema (one row per (benchmark, method) cell, averaged across
seeds):

    benchmark, method, exact_volume, estimated_volume, ratio,
    empirical_coverage

Methods that don't compute a reach set (RS, SAVER) are excluded.
ProbStar and Hashemi-clipping baselines on Exp 3 are NOT in our fake
data (they don't record volumes there); those rows surface only if the
input CSVs grow such columns later, in which case the loader picks them
up automatically.

Required LaTeX packages: ``booktabs``, ``multirow`` (the latter is used
to stack rows under a benchmark heading).
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


# ---------- Display ordering and labels -------------------------------

# Method ordering inside a benchmark block: looser first, tighter last,
# so the eye finds "ours" at the bottom (the punchline for each benchmark).
METHODS_ORDER = ["hyperrect", "ellipsoid", "gmm", "flow"]
METHOD_DISPLAY = {
    "hyperrect": "Hyper-rectangle",
    "ellipsoid": "Ellipsoid",
    "gmm":       "GMM",
    "flow":      "Flow (ours)",
}

# Benchmarks displayed in this table. The synthetic 1-Lipschitz net is
# split per-dim because the volume scales differently with d.
BENCHMARK_DISPLAY = {
    "3d_banana":            "3D banana (nonlin.)",
    "5d_1lip_id":           "5D 1-Lip identity",
    "10d_1lip_id":          "10D 1-Lip identity",
    "20d_1lip_id":          "20D 1-Lip identity",
    "geo_axis_aligned":     "Geo-axis identity",
    "geo_rotated":          "Geo-rotated identity",
}

BENCHMARK_ORDER = list(BENCHMARK_DISPLAY.keys())


# ---------- Number formatting ----------------------------------------

def _fmt_ratio(v: float) -> str:
    if v != v:  # NaN
        return "--"
    if v >= 100.0:
        return f"{v:.0f}"
    if v >= 10.0:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _fmt_volume(v: float) -> str:
    if v != v or v <= 0:
        return "--"
    if v >= 1e5:
        return f"{v:.2e}"
    if v >= 1.0:
        return f"{v:.1f}"
    return f"{v:.3f}"


def _fmt_cov(v: float) -> str:
    if v != v:
        return "--"
    return f"{v:.3f}"


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]  # drop NaN
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def _safe_float(v: object, default: float = float("nan")) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# ---------- CSV loaders ----------------------------------------------
#
# Each loader returns a list of dict rows with the canonical keys
# ``(benchmark, method, exact_volume, estimated_volume, ratio,
# empirical_coverage)``. Rows from all loaders are then aggregated
# (mean across seeds) by (benchmark, method).


def _load_ablation_score(csv_dir: Path) -> list[dict]:
    """Read ``ablation_score.csv`` — one row per (network, score, seed).

    Reconstructs ``exact_volume = volume / volume_ratio`` (since the
    volume floor is the (1-α) reach-set, which cancels in the ratio
    column). For closed-form linear nets this matches the analytical
    floor; for the banana it matches the Star-union ground truth.
    """
    rows = read_csv_rows(csv_dir / "ablation_score.csv")
    out = []
    for r in rows:
        net = (r.get("network") or "").strip()
        score = (r.get("score") or "").strip()
        if not net or not score:
            continue
        vol = _safe_float(r.get("volume"))
        ratio = _safe_float(r.get("volume_ratio"))
        cov = _safe_float(r.get("empirical_coverage"))
        exact = vol / ratio if ratio and ratio > 0 else float("nan")
        out.append({
            "benchmark": net,
            "method": score,
            "exact_volume": exact,
            "estimated_volume": vol,
            "ratio": ratio,
            "empirical_coverage": cov,
        })
    return out


def _load_exp3_synthetic(csv_dir: Path) -> list[dict]:
    """Per-dim flow rows for the d-D 1-Lipschitz identity net.

    Each row's benchmark is keyed as ``{d}d_1lip_id`` so it lines up with
    ``ablation_score.csv``'s ``5d_1lip_id`` / ``10d_1lip_id`` /
    ``20d_1lip_id`` rows.
    """
    rows = read_csv_rows(csv_dir / "exp3_synthetic_ours.csv")
    out = []
    for r in rows:
        try:
            d = int(r.get("dim", "0"))
        except (ValueError, TypeError):
            continue
        if d <= 0:
            continue
        vol = _safe_float(r.get("volume_estimate"))
        ratio = _safe_float(r.get("volume_ratio_vs_exact"))
        cov = _safe_float(r.get("coverage_empirical"))
        exact = vol / ratio if ratio and ratio > 0 else float("nan")
        out.append({
            "benchmark": f"{d}d_1lip_id",
            "method": "flow",
            "exact_volume": exact,
            "estimated_volume": vol,
            "ratio": ratio,
            "empirical_coverage": cov,
        })
    return out


def _load_exp3_banana(csv_dir: Path) -> list[dict]:
    """Flow rows on the 3D banana benchmark (nonlinear)."""
    rows = read_csv_rows(csv_dir / "exp3_3d_banana_ours.csv")
    out = []
    for r in rows:
        vol = _safe_float(r.get("volume_estimate"))
        ratio = _safe_float(r.get("volume_ratio_vs_exact"))
        cov = _safe_float(r.get("coverage_empirical"))
        exact = vol / ratio if ratio and ratio > 0 else float("nan")
        out.append({
            "benchmark": "3d_banana",
            "method": "flow",
            "exact_volume": exact,
            "estimated_volume": vol,
            "ratio": ratio,
            "empirical_coverage": cov,
        })
    return out


def _load_exp3_geo(csv_dir: Path) -> list[dict]:
    """Geo-transform rows. Network field is one of
    ``identity_axis_aligned`` / ``rotated``; we map them to short keys.
    """
    rows = read_csv_rows(csv_dir / "exp3_geo_transforms_ours.csv")
    name_map = {
        "identity_axis_aligned": "geo_axis_aligned",
        "rotated":               "geo_rotated",
    }
    out = []
    for r in rows:
        net = (r.get("network") or "").strip()
        bench = name_map.get(net)
        if bench is None:
            continue
        vol = _safe_float(r.get("volume_estimate"))
        ratio = _safe_float(r.get("volume_ratio_vs_exact"))
        cov = _safe_float(r.get("coverage_empirical"))
        exact = vol / ratio if ratio and ratio > 0 else float("nan")
        out.append({
            "benchmark": bench,
            "method": "flow",
            "exact_volume": exact,
            "estimated_volume": vol,
            "ratio": ratio,
            "empirical_coverage": cov,
        })
    return out


# Optional: pull volumes out of baseline CSVs IF they grow a
# ``volume_estimate`` column in the future. For now this is a no-op since
# the existing fake data's baseline CSVs don't record volumes.
def _load_baseline_volumes(csv_dir: Path) -> list[dict]:
    out = []
    for baseline_name in ("hashemi_clipping", "probstar"):
        for bench in ("3d_banana", "5d_1lip_id", "10d_1lip_id",
                      "20d_1lip_id"):
            path = csv_dir / f"baseline_{baseline_name}_{bench}.csv"
            rows = read_csv_rows(path)
            for r in rows:
                vol = _safe_float(r.get("volume_estimate"))
                if vol != vol:
                    continue
                ratio = _safe_float(r.get("volume_ratio"))
                cov = _safe_float(r.get("coverage_empirical"))
                exact = vol / ratio if ratio and ratio > 0 else float("nan")
                out.append({
                    "benchmark": bench,
                    "method": baseline_name,
                    "exact_volume": exact,
                    "estimated_volume": vol,
                    "ratio": ratio,
                    "empirical_coverage": cov,
                })
    return out


# ---------- Aggregation ----------------------------------------------

def _aggregate(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Average each cell across seeds. Keyed by (benchmark, method)."""
    bucket = defaultdict(lambda: {
        "exact_volume": [],
        "estimated_volume": [],
        "ratio": [],
        "empirical_coverage": [],
    })
    for r in rows:
        key = (r["benchmark"], r["method"])
        for col in ("exact_volume", "estimated_volume", "ratio",
                    "empirical_coverage"):
            bucket[key][col].append(_safe_float(r.get(col)))
    out = {}
    for key, cols in bucket.items():
        out[key] = {col: _mean(vs) for col, vs in cols.items()}
    return out


# ---------- Table assembly -------------------------------------------

def build_table(csv_dir: Path) -> str:
    rows = []
    rows += _load_ablation_score(csv_dir)
    rows += _load_exp3_synthetic(csv_dir)
    rows += _load_exp3_banana(csv_dir)
    rows += _load_exp3_geo(csv_dir)
    rows += _load_baseline_volumes(csv_dir)

    cells = _aggregate(rows)

    # Method order with optional baselines appended on the right end.
    extra_methods = sorted({
        m for (_, m) in cells.keys()
        if m not in METHODS_ORDER
    })
    method_order = METHODS_ORDER + extra_methods

    body = []
    body.append(r"% Auto-generated by tab_volumes.py.")
    body.append(r"% Required LaTeX packages: booktabs.")
    body.append(r"\begin{table}[t]")
    body.append(r"\centering")
    body.append(
        r"\caption{Reach-set volumes across Exp~3 benchmarks. For each "
        r"(benchmark, method) cell the table reports the exact (or "
        r"Star-union ground-truth) volume, the calibrated reach-set "
        r"volume estimate, the ratio (lower is tighter; 1.0 is the "
        r"floor), and the empirical coverage on held-out test points "
        r"($N=1000$ for our pipeline; same for baselines that bound a "
        r"reach box). Numbers are means across seeds. Bold method is "
        r"ours.}"
    )
    body.append(r"\label{tab:volumes}")
    body.append(r"\small")
    body.append(r"\begin{tabular}{ll rrrr}")
    body.append(r"\toprule")
    body.append(
        r"Benchmark & Method & "
        r"Exact vol. & Est.\ vol. & Ratio & Emp.\ cov. \\"
    )
    body.append(r"\midrule")

    first_block = True
    for bench in BENCHMARK_ORDER + sorted({
            b for (b, _) in cells.keys() if b not in BENCHMARK_ORDER}):
        block_rows = []
        for m in method_order:
            cell = cells.get((bench, m))
            if cell is None:
                continue
            block_rows.append((m, cell))
        if not block_rows:
            continue
        if not first_block:
            body.append(r"\midrule")
        first_block = False
        bench_disp = BENCHMARK_DISPLAY.get(bench, bench)
        for k, (m, cell) in enumerate(block_rows):
            method_disp = METHOD_DISPLAY.get(m, m)
            if m == "flow":
                method_disp = bold(method_disp)
            bench_cell = bench_disp if k == 0 else ""
            body.append(
                " & ".join([
                    bench_cell,
                    method_disp,
                    _fmt_volume(cell["exact_volume"]),
                    _fmt_volume(cell["estimated_volume"]),
                    _fmt_ratio(cell["ratio"]),
                    _fmt_cov(cell["empirical_coverage"]),
                ]) + r" \\"
            )

    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")
    body.append(r"\end{table}")
    return latex_document_wrapper("\n".join(body), caption="", label="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="tab_volumes.tex")
    args = parser.parse_args()
    output = args.output or (
        Path(__file__).resolve().parent / "tab_volumes.tex"
    )
    table_tex = build_table(args.csv_dir)
    write_table(output, table_tex)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
