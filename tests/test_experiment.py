"""Focused offline integration checks for the experiment runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import tabpack.experiment as experiment
from tabpack.core import ReducedConfig, binary_log_loss, binary_roc_auc
from tabpack.data import make_synthetic_fixture, prepare_data
from tabpack.models import OrdinaryMLP


@pytest.fixture
def synthetic_data():
    """Use only the explicit, offline 32/12/12 synthetic fixture."""

    return prepare_data(make_synthetic_fixture(), n_quantiles=16, random_state=0)


@pytest.fixture
def tiny_config():
    return ReducedConfig(
        seeds=(0,),
        member_count=2,
        max_selected=2,
        max_epochs=2,
        patience=2,
        batch_size=8,
        baseline_depth=1,
        baseline_width=4,
        baseline_dropout=0.0,
        baseline_learning_rate=0.01,
        baseline_weight_decay=0.0001,
        heterogeneous_depths=(1, 2),
        heterogeneous_widths=(4, 8),
        heterogeneous_dropouts=(0.0,),
        heterogeneous_learning_rate_range=(0.003, 0.003),
        heterogeneous_weight_decay_range=(0.0001, 0.0001),
    )


def _result_paths(root: Path) -> dict[str, Path]:
    return {
        method: root / f"{method}-seed-0" / "result.json"
        for method in experiment.METHODS
    }


def _load_results(root: Path) -> dict[str, dict]:
    return {
        method: json.loads(path.read_text(encoding="utf-8"))
        for method, path in _result_paths(root).items()
    }


def _replay_checkpoint(
    checkpoint: Path, data, *, expected_method: str
) -> tuple[np.ndarray, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["schema"] == "tabpack-churn.inference-checkpoint/v1"
    assert payload["method"] == expected_method
    modules = []
    for member in payload["members"]:
        module = OrdinaryMLP(
            int(member["input_dim"]),
            depth=int(member["depth"]),
            width=int(member["width"]),
            dropout=float(member["dropout"]),
            device="cpu",
            dtype=torch.float32,
        )
        module.load_state_dict(member["state_dict"])
        module.eval()
        modules.append(module)
    probabilities = experiment._probabilities(
        modules, data.test, torch.device("cpu")
    )
    return probabilities, payload


def _metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": binary_roc_auc(y.tolist(), probabilities.tolist()),
        "log_loss": binary_log_loss(y.tolist(), probabilities.tolist()),
        "accuracy": float(np.mean((probabilities >= 0.5) == y)),
    }


def _assert_finite_metrics(metrics: dict[str, float]) -> None:
    assert set(metrics) == {"roc_auc", "log_loss", "accuracy"}
    assert all(np.isfinite(value) for value in metrics.values())
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["log_loss"] >= 0.0


def test_synthetic_all_methods_checkpoint_replay_and_freeze(
    synthetic_data, tiny_config, tmp_path, monkeypatch
):
    """Train every method, freeze selection, and replay its saved weights."""

    data = synthetic_data
    assert data.dataset_id == "synthetic_smoke_v1"
    assert [len(data.train), len(data.val), len(data.test)] == [32, 12, 12]
    assert all(np.isfinite(features).all() for features in (data.train, data.val, data.test))
    assert data.preprocessor.metadata["fit_row_count"] == 32
    assert "violet" not in data.preprocessor.metadata["categorical"]["categories"][0]
    unknown_rows = [index for index, row_id in enumerate(data.source_indices["val"]) if row_id in {32, 37}]
    assert unknown_rows
    category_names = data.preprocessor.feature_names
    violet_columns = [index for index, name in enumerate(category_names) if "violet" in name]
    assert not violet_columns
    # Unknown held-out categories are ignored by the train-fitted encoder, so
    # their categorical block must be all zeros rather than extending it.
    assert np.all(data.val[unknown_rows, 2:] == 0.0)

    events: list[tuple[str, str]] = []
    original_train = experiment._train_method
    original_probabilities = experiment._probabilities
    original_metrics = experiment._single_metrics

    def train_with_freeze_probe(method, *args, **kwargs):
        frozen = original_train(method, *args, **kwargs)
        assert frozen.selection["frozen"] is True
        events.append(("freeze", method))
        return frozen

    def probabilities_with_test_probe(modules, features, device):
        if features is data.test:
            events.append(("test_predict", str(len(events))))
        return original_probabilities(modules, features, device)

    def metrics_with_test_probe(labels, probabilities):
        if labels is data.y_test:
            events.append(("test_metrics", str(len(events))))
        return original_metrics(labels, probabilities)

    monkeypatch.setattr(experiment, "_train_method", train_with_freeze_probe)
    monkeypatch.setattr(experiment, "_probabilities", probabilities_with_test_probe)
    monkeypatch.setattr(experiment, "_single_metrics", metrics_with_test_probe)

    root = experiment.run_comparison(
        data,
        config=tiny_config,
        output=tmp_path / "run-a",
        requested_device="cpu",
    )
    results = _load_results(root)
    assert set(results) == set(experiment.METHODS)
    assert [item[0] for item in events].count("freeze") == 3
    assert [item[0] for item in events].count("test_predict") == 3
    assert [item[0] for item in events].count("test_metrics") == 3
    for method in experiment.METHODS:
        freeze_index = events.index(("freeze", method))
        test_index = next(
            index for index, item in enumerate(events)
            if item[0] == "test_predict" and index > freeze_index
        )
        assert freeze_index < test_index

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data"]["dataset_id"] == "synthetic_smoke_v1"
    assert manifest["expected_seeds"] == [0]
    assert manifest["expected_methods"] == list(experiment.METHODS)
    for method, result in results.items():
        assert result["status"] == "complete"
        assert result["data"] == "synthetic_smoke_v1"
        _assert_finite_metrics(result["metrics"]["validation"])
        _assert_finite_metrics(result["metrics"]["test"])
        assert result["selection"]["frozen"] is True
        assert 1 <= result["selection"]["selected_snapshot_count"] <= 2
        assert 1 <= result["training"]["epochs_completed"] <= 2
        probabilities, payload = _replay_checkpoint(
            root / result["artifacts"]["checkpoint"], data, expected_method=method
        )
        assert np.isfinite(probabilities).all()
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
        replay_metrics = _metrics(data.y_test, probabilities)
        for name, value in replay_metrics.items():
            assert result["metrics"]["test"][name] == pytest.approx(
                value, rel=1e-7, abs=1e-8
            )
        assert len(payload["members"]) == result["selection"]["selected_snapshot_count"]


def test_cpu_replay_is_deterministic(synthetic_data, tiny_config, tmp_path):
    """Fresh CPU runs retain the same selected snapshots and numeric outcomes."""

    first = experiment.run_comparison(
        synthetic_data,
        config=tiny_config,
        output=tmp_path / "first",
        requested_device="cpu",
    )
    second = experiment.run_comparison(
        synthetic_data,
        config=tiny_config,
        output=tmp_path / "second",
        requested_device="cpu",
    )
    first_results = _load_results(first)
    second_results = _load_results(second)
    for method in experiment.METHODS:
        left = first_results[method]
        right = second_results[method]
        assert left["selection"]["selected_snapshot_ids"] == right["selection"]["selected_snapshot_ids"]
        assert left["selection"] == right["selection"]
        assert left["training"] == right["training"]
        for split in ("validation", "test"):
            for metric in ("roc_auc", "log_loss", "accuracy"):
                assert left["metrics"][split][metric] == pytest.approx(
                    right["metrics"][split][metric], rel=1e-7, abs=1e-8
                )
        left_probabilities, _ = _replay_checkpoint(
            first / first_results[method]["artifacts"]["checkpoint"],
            synthetic_data,
            expected_method=method,
        )
        right_probabilities, _ = _replay_checkpoint(
            second / second_results[method]["artifacts"]["checkpoint"],
            synthetic_data,
            expected_method=method,
        )
        np.testing.assert_allclose(left_probabilities, right_probabilities, rtol=1e-7, atol=1e-8)


def test_non_empty_output_is_refused(synthetic_data, tiny_config, tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    sentinel = occupied / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        experiment.run_comparison(
            synthetic_data,
            config=tiny_config,
            output=occupied,
            requested_device="cpu",
            methods=("mlp",),
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (occupied / "manifest.json").exists()


def test_report_rejects_incomplete_matrix_and_manifest_tampering(
    synthetic_data, tiny_config, tmp_path
):
    root = experiment.run_comparison(
        synthetic_data,
        config=tiny_config,
        output=tmp_path / "report-source",
        requested_device="cpu",
    )
    result_path = _result_paths(root)["mlp"]
    result_text = result_path.read_text(encoding="utf-8")
    result_path.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        experiment.report_comparison(root, tmp_path / "incomplete-report")
    result_path.write_text(result_text, encoding="utf-8")

    manifest_path = root / "manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    manifest["expected_seeds"] = [1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected seed"):
        experiment.report_comparison(root, tmp_path / "matrix-tampered-report")
    manifest_path.write_text(manifest_text, encoding="utf-8")

    manifest = json.loads(manifest_text)
    manifest["comparison_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash|comparison"):
        experiment.report_comparison(root, tmp_path / "hash-tampered-report")
