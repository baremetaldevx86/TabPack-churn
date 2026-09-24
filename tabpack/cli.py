"""Command-line boundary for preflight, acquisition, smoke, run, and report."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from .core import ReducedConfig
from .data import ChurnDataError, acquire_churn, load_churn, prepare_data
from .experiment import (
    METHODS,
    environment_record,
    report_comparison,
    resolve_device,
    run_comparison,
    smoke_comparison,
)


def _json_print(value: object) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


def _preflight(args: argparse.Namespace) -> int:
    device, fallback = resolve_device(args.device)
    disk = shutil.disk_usage(Path.cwd())
    record = environment_record(device, requested_device=args.device, fallback=fallback)
    record.update(
        {
            "numpy": np.__version__,
            "free_disk_bytes": disk.free,
            "working_directory": str(Path.cwd()),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    )
    if args.data_dir:
        root = Path(args.data_dir)
        record["data_directory_exists"] = root.is_dir()
        record["data_directory"] = root.as_posix()
    _json_print(record)
    return 0


def _load_prepared(data_dir: str) -> object:
    raw = load_churn(data_dir)
    return prepare_data(raw)


def _dry_run(args: argparse.Namespace) -> int:
    device, fallback = resolve_device(args.device)
    prepared = _load_prepared(args.data_dir)
    config = ReducedConfig()
    _json_print(
        {
            "status": "ready",
            "dataset_id": prepared.dataset_id,
            "dataset_sha256": prepared.dataset_sha256,
            "preprocessing_sha256": prepared.preprocessor.fingerprint,
            "shapes": {
                "train": list(prepared.train.shape),
                "val": list(prepared.val.shape),
                "test": list(prepared.test.shape),
            },
            "config": config.to_dict(),
            "device": str(device),
            "fallback": fallback,
        }
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    prepared = _load_prepared(args.data_dir)
    config = ReducedConfig()
    output = run_comparison(
        prepared,
        config=config,
        output=args.output,
        requested_device=args.device,
        experiment_id="churn-reduced-v1",
        methods=METHODS,
    )
    _json_print({"status": "complete", "run_directory": str(output)})
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="check interpreter, PyTorch, device, and disk")
    preflight.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    preflight.add_argument("--data-dir")
    preflight.set_defaults(handler=_preflight)

    fetch = subparsers.add_parser("fetch-data", help="download and extract the pinned Churn archive")
    fetch.add_argument("--data-dir", required=True)
    fetch.set_defaults(handler=lambda args: (acquire_churn(args.data_dir), _json_print({"status": "complete", "data_dir": args.data_dir}))[1] or 0)

    dry = subparsers.add_parser("dry-run", help="validate real Churn and preprocessing without training")
    dry.add_argument("--data-dir", required=True)
    dry.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    dry.set_defaults(handler=_dry_run)

    smoke = subparsers.add_parser("smoke", help="run the tiny synthetic all-method engineering smoke")
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    smoke.set_defaults(handler=lambda args: (smoke_comparison(args.output, requested_device=args.device), _json_print({"status": "complete", "run_directory": args.output}))[1] or 0)

    run = subparsers.add_parser("run", help="run the three-method canonical Churn comparison")
    run.add_argument("--data-dir", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    run.set_defaults(handler=_run)

    report = subparsers.add_parser("report", help="aggregate an existing run without retraining")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--output", required=True)
    report.set_defaults(handler=lambda args: (report_comparison(args.run_dir, args.output), _json_print({"status": "complete", "report_directory": args.output}))[1] or 0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ChurnDataError, FileExistsError, RuntimeError, ValueError, FloatingPointError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
