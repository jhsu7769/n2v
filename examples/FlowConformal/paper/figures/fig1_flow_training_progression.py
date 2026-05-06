"""Figure 1 — Flow training progression overlay.

Thin wrapper that copies (or re-uses) the existing pre-rendered overlay
in ``flow_matching_training/overlay.png``. The original heavy
training-progression script (``flow_matching_training/fig_training_progression.py``)
is preserved alongside; it produces the same overlay from scratch.

Run with ``--regenerate`` to invoke the full training pipeline (slow:
needs torch + sinkhorn coupling + Star propagation).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import add_common_args  # noqa: E402

HERE = Path(__file__).resolve().parent
SOURCE_PNG = HERE / "flow_matching_training" / "overlay.png"
SOURCE_SCRIPT = HERE / "flow_matching_training" / "fig_training_progression.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="fig1_flow_training_progression.png")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Re-train the flow snapshots and re-render the overlay.",
    )
    args = parser.parse_args()
    output = args.output or (HERE / "fig1_flow_training_progression.png")

    if args.regenerate:
        if not SOURCE_SCRIPT.exists():
            print(f"ERROR: cannot find {SOURCE_SCRIPT}", file=sys.stderr)
            sys.exit(1)
        # The underlying script writes overlay.png next to itself.
        subprocess.check_call([sys.executable, str(SOURCE_SCRIPT)])

    if not SOURCE_PNG.exists():
        print(f"ERROR: expected pre-rendered overlay at {SOURCE_PNG}.", file=sys.stderr)
        print("       Re-run with --regenerate to produce it.", file=sys.stderr)
        sys.exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SOURCE_PNG, output)
    print(f"Wrote {output} (copied from {SOURCE_PNG})")


if __name__ == "__main__":
    main()
