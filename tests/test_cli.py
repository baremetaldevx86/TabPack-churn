"""Focused CLI integration checks that stay offline and synthetic."""

from __future__ import annotations

import json

from tabpack import cli
from tabpack.experiment import METHODS


def test_cli_preflight_cpu_is_json(capsys):
    assert cli.main(["preflight", "--device", "cpu"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["requested_device"] == "cpu"
    assert output["selected_device"] == "cpu"
    assert output["fallback"] is False
    assert output["cuda_available"] in (True, False)


def test_cli_synthetic_smoke_and_report(tmp_path, capsys):
    run_directory = tmp_path / "cli-smoke"
    assert cli.main(
        ["smoke", "--device", "cpu", "--output", str(run_directory)]
    ) == 0
    smoke_output = json.loads(capsys.readouterr().out)
    assert smoke_output["status"] == "complete"
    assert set(path.name for path in run_directory.iterdir()) >= {
        "manifest.json",
        *[f"{method}-seed-0" for method in METHODS],
    }

    report_directory = tmp_path / "cli-report"
    assert cli.main(
        [
            "report",
            "--run-dir",
            str(run_directory),
            "--output",
            str(report_directory),
        ]
    ) == 0
    report_output = json.loads(capsys.readouterr().out)
    assert report_output == {
        "report_directory": str(report_directory),
        "status": "complete",
    }
    summary = json.loads((report_directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] is True
    assert set(summary["groups"]) == set(METHODS)
    assert (report_directory / "per_seed.csv").is_file()
    assert (report_directory / "report.md").is_file()


def test_cli_smoke_refuses_non_empty_output(tmp_path, capsys):
    output = tmp_path / "occupied"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    assert cli.main(["smoke", "--device", "cpu", "--output", str(output)]) == 2
    assert "not empty" in capsys.readouterr().err
    assert sentinel.read_text(encoding="utf-8") == "keep"
