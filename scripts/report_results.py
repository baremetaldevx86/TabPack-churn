#!/usr/bin/env python3
"""Write JSON, CSV, and Markdown reports from a completed experiment run.

Usage from the repository root::

    python scripts/report_results.py \
        --run-dir runs/churn-reduced-committed \
        --output reports/churn-reduced-committed

The script performs no training and no data acquisition. It validates the run
matrix and checkpoint digests through ``tabpack.experiment.report_comparison``
before writing ``summary.json``, ``per_seed.csv``, and ``report.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write JSON, CSV, and Markdown reports from a completed run."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed run directory containing manifest.json and result files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for summary.json, per_seed.csv, and report.md.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _repository_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from tabpack.experiment import report_comparison

    try:
        output = report_comparison(args.run_dir, args.output)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {output / 'summary.json'}")
    print(f"Wrote {output / 'per_seed.csv'}")
    print(f"Wrote {output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
