"""Regenerate every paper figure and table in one go.

Usage:
    python -m examples.FlowConformal.paper.regenerate_all --csv-dir <real_results_dir>

``--csv-dir`` is REQUIRED; there is no fake-data fallback. The path is
forwarded verbatim to every figure / table script, each of which also
requires ``--csv-dir``.

Discovery: scripts are auto-discovered by glob (``tab*.py`` and
``fig*.py`` under ``tables/`` and ``figures/`` respectively). The
explicit ``ORDERED_PREFIXES`` list below pins the lexicographic order
of high-priority artefacts so the most-cited ones run first; any new
script that doesn't match a prefix runs in alphabetical order after
the priority block.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Priority prefixes — these run first in this order. Anything else under
# tables/ or figures/ matching ``tab*.py`` / ``fig*.py`` runs after, in
# alphabetical order.
ORDERED_PREFIXES = (
    "tab1_",
    "tab2_",
    "tab3_",
    "fig1_",
    "fig2_",
    "fig3_",
    "fig4_",
    "fig4a_",
    "fig5_",
    "fig5a_",
    "fig5b_",
    "fig6_",
    "fig6a_",
    "fig6b_",
    "fig6c_",
    "fig7_",
)


def _discover() -> list[Path]:
    """Return ordered list of generation scripts to run.

    Order: priority prefixes first (in declaration order), then anything
    else lexicographically. Skips ``_*.py`` (helpers) and
    ``flow_matching_training/`` (the heavyweight training script).
    """
    candidates: list[Path] = []
    for sub in ("tables", "figures"):
        for p in sorted((HERE / sub).glob("*.py")):
            if p.name.startswith("_"):
                continue
            candidates.append(p)

    by_prefix: dict[str, list[Path]] = {pref: [] for pref in ORDERED_PREFIXES}
    leftovers: list[Path] = []
    for p in candidates:
        matched = False
        for pref in ORDERED_PREFIXES:
            if p.name.startswith(pref):
                by_prefix[pref].append(p)
                matched = True
                break
        if not matched:
            leftovers.append(p)

    ordered: list[Path] = []
    for pref in ORDERED_PREFIXES:
        ordered.extend(by_prefix[pref])
    ordered.extend(leftovers)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-dir",
        type=Path,
        required=True,
        help="Directory holding the input CSVs (REQUIRED — no default, no fake-data fallback).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use (default: current).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print discovered scripts and exit (smoke).",
    )
    parser.add_argument(
        "--skip", nargs="*", default=(),
        help="Filenames to skip (e.g. fig7_banana_score_geometries.py).",
    )
    args = parser.parse_args()

    scripts = _discover()
    if args.skip:
        scripts = [s for s in scripts if s.name not in set(args.skip)]
    if args.list:
        for s in scripts:
            print(s.relative_to(HERE))
        return

    failures: list[str] = []
    for script in scripts:
        cmd = [args.python, str(script), "--csv-dir", str(args.csv_dir)]
        print(f"-- running {script.relative_to(HERE)}")
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"   FAILED: {e}", file=sys.stderr)
            failures.append(script.name)

    if failures:
        print(f"\n{len(failures)} script(s) failed: {failures}", file=sys.stderr)
        sys.exit(1)
    print(f"\nAll {len(scripts)} scripts succeeded.")


if __name__ == "__main__":
    main()
