"""Checks for the standalone report script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tabpack.experiment import smoke_comparison


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "report_results.py"
    spec = importlib.util.spec_from_file_location("report_results_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_script_writes_json_csv_and_markdown(tmp_path):
    script = _load_script()
    run_dir = smoke_comparison(tmp_path / "run", requested_device="cpu")
    output = tmp_path / "report"

    assert script.main(["--run-dir", str(run_dir), "--output", str(output)]) == 0
    assert (output / "summary.json").is_file()
    assert (output / "per_seed.csv").is_file()
    assert (output / "report.md").is_file()
