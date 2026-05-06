"""LaTeX-table helpers shared across paper-table scripts."""

from __future__ import annotations

import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent
if str(PAPER_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR.parent.parent))

from FlowConformal.paper._common import (  # noqa: F401  (re-exported)
    BENCHMARK_DISPLAY,
    EXP1_BENCHMARKS,
    EXP1_SOUND_VERIFIERS,
    EXP2_BENCHMARKS,
    EXP2_SOUND_VERIFIERS,
    METHOD_DISPLAY,
    PAPER_DIR,
    SOLVED_VERDICTS,
    VERDICT_ORDER,
    add_common_args,
    count_verdicts,
    mean_wall_clock,
    normalize_verdict,
    percent_solved,
    read_csv_no_header,
    read_csv_rows,
)


# ---- LaTeX cell formatting ----

def fmt_int(n: int) -> str:
    return f"{n}"


def fmt_pct(p: float, ndigits: int = 1) -> str:
    return f"{p:.{ndigits}f}"


def fmt_seconds(s: float) -> str:
    if s == 0.0:
        return "--"
    if s >= 100.0:
        return f"{s:.0f}"
    return f"{s:.1f}"


def latex_escape(text: str) -> str:
    """Minimal LaTeX-safe escape for table cells."""
    return (
        text.replace("\\", r"\\")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("#", r"\#")
            .replace("$", r"\$")
    )


def bold(s: str) -> str:
    return r"\textbf{" + s + "}"


def italic(s: str) -> str:
    return r"\textit{" + s + "}"


def latex_document_wrapper(table_body: str, caption: str, label: str) -> str:
    """Standalone LaTeX wrapper so the user can compile the .tex directly."""
    preamble = (
        "% Auto-generated table. Compile with `pdflatex` or include in your manuscript.\n"
        "% Required packages: booktabs, multirow.\n"
    )
    return preamble + table_body + "\n"


def write_table(out_path: Path, table_tex: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table_tex)
