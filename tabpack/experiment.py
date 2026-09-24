"""Coordinator-owned training and reporting for the reduced Churn comparison.

This module intentionally keeps the orchestration explicit.  It trains each
method on the same train/validation arrays, freezes validation selection, and
only then evaluates the frozen predictor once on the test split.  It does not
implement interrupted-training resume and never imports the upstream project.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .core import (
    Candidate,
    ReducedConfig,
    aggregate_results,
    binary_log_loss,
    binary_roc_auc,
    canonical_json,
    greedy_select,
    named_seed,
    retain_best_selection,
    sample_members,
)
from .data import PreparedData
from .models import OrdinaryMLP, PackedMLP, packed_bce_loss
from .optim import make_optimizer

METHODS = (
    "mlp",
    "independent_mlp_ensemble",
    "packed_heterogeneous_ensemble",
)
PROTOCOL_ID = "reduced-churn-v1"
REFERENCE_COMMIT = "05a89e21b955f12de84889d662e15ca534019aaa"


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _clone_state(module: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _load_state(module: nn.Module, state: Mapping[str, Tensor], device: torch.device) -> nn.Module:
    module.load_state_dict(state)
    return module.to(device)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def resolve_device(requested: str) -> tuple[torch.device, bool]:
    """Resolve ``cpu``, ``cuda`` or ``auto`` and report whether auto fell back."""

    if requested not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be cpu, cuda, or auto")
    if requested == "cpu":
        return torch.device("cpu"), False
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cpu"), True
    try:
        torch.zeros(1, device="cuda").add_(1)
        torch.cuda.synchronize()
    except Exception as exc:
        if requested == "cuda":
            raise RuntimeError(f"CUDA preflight allocation failed: {exc}") from exc
        return torch.device("cpu"), True
    return torch.device("cuda"), False


def environment_record(device: torch.device, *, requested_device: str, fallback: bool) -> dict[str, Any]:
    """Return a secret-free environment record suitable for a run manifest."""

    record: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "requested_device": requested_device,
        "selected_device": str(device),
        "fallback": fallback,
        "dtype": "float32",
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "tf32_matmul": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        "cpu_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        record.update(
            {
                "device_name": props.name,
                "device_capability": list(torch.cuda.get_device_capability(device)),
                "device_memory_bytes": props.total_memory,
            }
        )
    else:
        record.update({"device_name": None, "device_capability": None, "device_memory_bytes": None})
    return record


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_start(device: torch.device) -> float:
    _sync(device)
    return time.perf_counter()


def _timed_end(device: torch.device, start: float) -> float:
    _sync(device)
    return time.perf_counter() - start


def _batch_slices(size: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, size, batch_size):
        yield start, min(size, start + batch_size)


def _permutation(size: int, generator: torch.Generator) -> Tensor:
    return torch.randperm(size, generator=generator, device="cpu")


def _probabilities(
    modules: Sequence[OrdinaryMLP], x: np.ndarray, device: torch.device
) -> np.ndarray:
    """Evaluate independent logits and average probabilities in float64."""

    if not modules:
        raise ValueError("at least one frozen module is required")
    features = torch.as_tensor(x, dtype=torch.float32, device=device)
    outputs: list[np.ndarray] = []
    for module in modules:
        module.eval()
        with torch.no_grad():
            logits = module(features).detach().to(device="cpu", dtype=torch.float64)
            outputs.append(torch.sigmoid(logits).numpy())
    return np.mean(np.stack(outputs, axis=0, dtype=np.float64), axis=0, dtype=np.float64)


def _single_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": binary_roc_auc(y.tolist(), probabilities.tolist()),
        "log_loss": binary_log_loss(y.tolist(), probabilities.tolist()),
        "accuracy": float(np.mean((probabilities >= 0.5) == y)),
    }


def _new_ordinary(spec: Any, input_dim: int, seed: int, device: torch.device) -> OrdinaryMLP:
    _seed_everything(seed)
    return OrdinaryMLP(
        input_dim,
        depth=spec.depth,
        width=spec.width,
        dropout=spec.dropout,
        device=device,
        dtype=torch.float32,
    )


def _train_one(
    model: OrdinaryMLP,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[OrdinaryMLP, float, int, float, str]:
    features = torch.as_tensor(np.array(x, copy=True), dtype=torch.float32, device=device)
    labels = torch.as_tensor(np.array(y, copy=True), dtype=torch.float32, device=device)
    permutation_generator = torch.Generator(device="cpu").manual_seed(
        named_seed(seed, "permutation", "ordinary")
    )
    best_auc = float("-inf")
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    bad_epochs = 0
    epochs_completed = 0
    fit_start = _timed_start(device)
    for epoch in range(max_epochs):
        model.train()
        order = _permutation(len(features), permutation_generator)
        for start, stop in _batch_slices(len(order), batch_size):
            indices = order[start:stop].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features[indices])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels[indices])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
        probabilities = _probabilities([model], val_x, device)
        auc = binary_roc_auc(val_y.tolist(), probabilities.tolist())
        epochs_completed = epoch + 1
        if auc > best_auc:
            best_auc = auc
            best_state = _clone_state(model)
            best_epoch = epoch + 1
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    fit_time = _timed_end(device, fit_start)
    if best_state is None:
        raise RuntimeError("ordinary model did not produce a validation checkpoint")
    _load_state(model, best_state, device).eval()
    return model, best_auc, epochs_completed, fit_time, "patience" if bad_epochs >= patience else "max_epochs", best_epoch


@dataclass
class FrozenMethod:
    modules: list[OrdinaryMLP]
    validation: dict[str, float]
    training: dict[str, Any]
    selection: dict[str, Any]
    fit_wall_s: float


def _train_mlp(data: PreparedData, config: ReducedConfig, seed: int, device: torch.device) -> FrozenMethod:
    spec = sample_members(config, seed, mode="homogeneous")[0]
    model = _new_ordinary(spec, data.train.shape[1], spec.streams["initialization"], device)
    optimizer = make_optimizer(model, lr=spec.learning_rate, weight_decay=spec.weight_decay)
    torch.manual_seed(spec.streams["dropout"])
    model, val_auc, epochs, fit_time, reason, best_epoch = _train_one(
        model, optimizer, data.train, data.y_train, data.val, data.y_val,
        seed=spec.streams["permutation"], batch_size=config.batch_size,
        max_epochs=config.max_epochs, patience=config.patience, device=device,
    )
    probs = _probabilities([model], data.val, device)
    return FrozenMethod(
        [model], _single_metrics(data.y_val, probs),
        {"epochs_completed": epochs, "stop_reason": reason, "base_member_count": 1},
        {"policy": "single_best_validation_auc", "frozen": True, "selected_snapshot_count": 1,
         "selected_snapshot_ids": [f"member-0000@epoch-{best_epoch:04d}"]}, fit_time,
    )


def _train_homogeneous(
    data: PreparedData, config: ReducedConfig, seed: int, device: torch.device
) -> FrozenMethod:
    specs = sample_members(config, seed, mode="homogeneous")
    models: list[OrdinaryMLP] = []
    optimizers: list[torch.optim.Optimizer] = []
    generators: list[torch.Generator] = []
    for spec in specs:
        model = _new_ordinary(spec, data.train.shape[1], spec.streams["initialization"], device)
        models.append(model)
        optimizers.append(make_optimizer(model, lr=spec.learning_rate, weight_decay=spec.weight_decay))
        generators.append(
            torch.Generator(device="cpu").manual_seed(spec.streams["permutation"])
        )
    features = torch.as_tensor(np.array(data.train, copy=True), dtype=torch.float32, device=device)
    labels = torch.as_tensor(np.array(data.y_train, copy=True), dtype=torch.float32, device=device)
    best_states: list[dict[str, Tensor]] | None = None
    best_auc = float("-inf")
    best_epoch = 0
    bad_epochs = 0
    epochs_completed = 0
    fit_start = _timed_start(device)
    for epoch in range(config.max_epochs):
        for model, optimizer, generator in zip(models, optimizers, generators):
            model.train()
            order = _permutation(len(features), generator)
            for start, stop in _batch_slices(len(order), config.batch_size):
                indices = order[start:stop].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(features[indices])
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels[indices])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite homogeneous loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
        val_probabilities = _probabilities(models, data.val, device)
        auc = binary_roc_auc(data.y_val.tolist(), val_probabilities.tolist())
        epochs_completed = epoch + 1
        if auc > best_auc:
            best_auc = auc
            best_states = [_clone_state(model) for model in models]
            best_epoch = epoch + 1
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= config.patience:
            break
    fit_time = _timed_end(device, fit_start)
    if best_states is None:
        raise RuntimeError("homogeneous ensemble did not produce a validation checkpoint")
    for model, state in zip(models, best_states):
        _load_state(model, state, device).eval()
    probabilities = _probabilities(models, data.val, device)
    return FrozenMethod(
        models,
        _single_metrics(data.y_val, probabilities),
        {"epochs_completed": epochs_completed, "stop_reason": "patience" if bad_epochs >= config.patience else "max_epochs",
         "base_member_count": len(models)},
        {"policy": "all_members_validation_checkpoint", "frozen": True,
         "selected_snapshot_count": len(models), "selected_snapshot_ids": [f"member-{i:04d}@epoch-{best_epoch:04d}" for i in range(len(models))]},
        fit_time,
    )


def _candidate_from_model(
    model: OrdinaryMLP, *, member_id: str, epoch: int, probabilities: np.ndarray
) -> Candidate:
    return Candidate(
        candidate_id=f"{member_id}@epoch-{epoch:04d}",
        member_id=member_id,
        source_epoch=epoch,
        probabilities=tuple(float(value) for value in probabilities.tolist()),
    )


def _train_packed(
    data: PreparedData, config: ReducedConfig, seed: int, device: torch.device
) -> FrozenMethod:
    specs = sample_members(config, seed, mode="heterogeneous")
    _seed_everything(named_seed(seed, "packed", "initialization"))
    packed = PackedMLP(
        data.train.shape[1],
        depths=[spec.depth for spec in specs],
        widths=[spec.width for spec in specs],
        dropouts=[spec.dropout for spec in specs],
        device=device,
        dtype=torch.float32,
    )
    optimizer = make_optimizer(
        packed,
        lr=[spec.learning_rate for spec in specs],
        weight_decay=[spec.weight_decay for spec in specs],
    )
    torch.manual_seed(named_seed(seed, "packed", "dropout"))
    features = torch.as_tensor(np.array(data.train, copy=True), dtype=torch.float32, device=device)
    labels = torch.as_tensor(np.array(data.y_train, copy=True), dtype=torch.float32, device=device)
    generators = [
        torch.Generator(device="cpu").manual_seed(spec.streams["permutation"])
        for spec in specs
    ]
    selected_result = None
    selected_models: dict[str, OrdinaryMLP] = {}
    best_result = None
    best_models: dict[str, OrdinaryMLP] = {}
    bad_epochs = 0
    epochs_completed = 0
    fit_start = _timed_start(device)
    for epoch in range(config.max_epochs):
        packed.train()
        orders = [_permutation(len(features), generator) for generator in generators]
        for start, stop in _batch_slices(len(features), config.batch_size):
            batch_indices = [order[start:stop] for order in orders]
            indices = torch.stack(batch_indices).to(device)
            packed_x = features[indices]
            packed_y = labels[indices]
            optimizer.zero_grad(set_to_none=True)
            logits = packed(packed_x)
            loss, _ = packed_bce_loss(logits, packed_y)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite packed loss at epoch {epoch}")
            loss.backward()
            optimizer.step()

        packed.eval()
        with torch.no_grad():
            validation_logits = packed(torch.as_tensor(data.val, dtype=torch.float32, device=device))
            validation_probabilities = torch.sigmoid(
                validation_logits.detach().to(device="cpu", dtype=torch.float64)
            ).numpy()
        current_models = {
            f"member-{index:04d}": packed.extract_member(index).eval()
            for index in range(len(specs))
        }
        current_candidates = [
            _candidate_from_model(
                current_models[f"member-{index:04d}"],
                member_id=f"member-{index:04d}", epoch=epoch + 1,
                probabilities=validation_probabilities[index],
            )
            for index in range(len(specs))
        ]
        prior_candidates = [] if selected_result is None else list(selected_result.selected)
        candidate_pool = prior_candidates + current_candidates
        candidate_models = dict(selected_models)
        candidate_models.update(
            {
                candidate.candidate_id: current_models[candidate.member_id]
                for candidate in current_candidates
            }
        )
        proposed = greedy_select(data.y_val.tolist(), candidate_pool, max_size=config.max_selected)
        selected_result = retain_best_selection(selected_result, proposed)
        selected_models = {candidate.candidate_id: candidate_models[candidate.candidate_id]
                           for candidate in selected_result.selected}
        epochs_completed = epoch + 1
        if best_result is None or selected_result.validation_auc > best_result.validation_auc:
            best_result = selected_result
            best_models = dict(selected_models)
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= config.patience:
            break
    fit_time = _timed_end(device, fit_start)
    if best_result is None or not best_models:
        raise RuntimeError("packed ensemble did not produce a frozen selection")
    selected_modules = [best_models[candidate.candidate_id] for candidate in best_result.selected]
    validation_probabilities = _probabilities(selected_modules, data.val, device)
    return FrozenMethod(
        selected_modules,
        _single_metrics(data.y_val, validation_probabilities),
        {"epochs_completed": epochs_completed,
         "stop_reason": "patience" if bad_epochs >= config.patience else "max_epochs",
         "base_member_count": len(specs), "logical_parameter_count": packed.logical_parameter_count,
         "stored_parameter_count": packed.stored_parameter_count,
         "sampled_members": [spec.to_dict() for spec in specs]},
        {"policy": "online_greedy_validation_auc", "frozen": True,
         "selected_snapshot_count": len(best_result.selected),
         "selected_snapshot_ids": list(best_result.selected_ids),
         "validation_auc": best_result.validation_auc,
         "stop_reason": best_result.stop_reason,
         "validation_evaluations": best_result.validation_evaluations,
         "candidate_order": list(best_result.candidate_order)},
        fit_time,
    )


def _checkpoint_payload(method: str, seed: int, frozen: FrozenMethod) -> dict[str, Any]:
    snapshot_ids = list(frozen.selection.get("selected_snapshot_ids", []))
    if len(snapshot_ids) != len(frozen.modules):
        raise ValueError("frozen selection IDs do not match checkpoint members")
    return {
        "schema": "tabpack-churn.inference-checkpoint/v1",
        "method": method,
        "seed": seed,
        "members": [
            {
                "candidate_id": snapshot_ids[index],
                "member_id": snapshot_ids[index].split("@", 1)[0],
                "source_epoch": int(snapshot_ids[index].rsplit("-", 1)[-1])
                if "epoch-" in snapshot_ids[index] else None,
                "input_dim": module.input_dim,
                "depth": module.depth,
                "width": module.width,
                "dropout": module.dropout,
                "output_dim": module.output_dim,
                "state_dict": _clone_state(module),
            }
            for index, module in enumerate(frozen.modules)
        ],
        "selection": frozen.selection,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _train_method(
    method: str, data: PreparedData, config: ReducedConfig, seed: int, device: torch.device
) -> FrozenMethod:
    if method == "mlp":
        return _train_mlp(data, config, seed, device)
    if method == "independent_mlp_ensemble":
        return _train_homogeneous(data, config, seed, device)
    if method == "packed_heterogeneous_ensemble":
        return _train_packed(data, config, seed, device)
    raise ValueError(f"unknown method: {method}")


def _run_identity(
    config: ReducedConfig,
    data: PreparedData,
    environment: Mapping[str, Any],
    methods: Sequence[str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "reference_commit": REFERENCE_COMMIT,
        "config": config.to_dict(),
        "dataset": {
            "dataset_id": data.dataset_id,
            "dataset_sha256": data.dataset_sha256,
            "preprocessing_sha256": data.preprocessor.fingerprint,
            "counts": {"train": len(data.train), "val": len(data.val), "test": len(data.test)},
        },
        "methods": list(methods),
        "source": dict(source),
        "environment_group": _sha256_json(environment),
    }


def run_comparison(
    data: PreparedData,
    *,
    config: ReducedConfig,
    output: str | Path,
    requested_device: str = "auto",
    methods: Sequence[str] = METHODS,
    experiment_id: str = "churn-reduced",
) -> Path:
    """Run all requested methods/seeds and write local result artifacts."""

    methods = tuple(methods)
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError(f"methods must be a non-empty subset of {METHODS}")
    if len(set(methods)) != len(methods):
        raise ValueError("methods must not contain duplicates")
    if config.dtype != "float32":
        raise ValueError("the reduced runner currently supports dtype='float32' only")
    device, fallback = resolve_device(requested_device)
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    environment = environment_record(device, requested_device=requested_device, fallback=fallback)
    source = {"local_commit": _git_commit(), "reference_commit": REFERENCE_COMMIT}
    identity = _run_identity(config, data, environment, methods, source)
    comparison_hash = _sha256_json(identity)
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "protocol_id": PROTOCOL_ID,
        "comparison_hash": comparison_hash,
        "expected_seeds": list(config.seeds),
        "expected_methods": list(methods),
        "config": config.to_dict(),
        "data": identity["dataset"],
        "source": source,
        "environment": environment,
        "environment_id": environment["environment_group"] if "environment_group" in environment else _sha256_json(environment),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Keep a stable environment identity in both manifest and records.
    environment_id = _sha256_json(environment)
    manifest["environment_id"] = environment_id
    _write_json(root / "manifest.json", manifest)
    for seed in config.seeds:
        for method in methods:
            run_id = f"{method}-seed-{seed}"
            run_dir = root / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            run_started = time.perf_counter()
            try:
                fit_started = _timed_start(device)
                frozen = _train_method(method, data, config, seed, device)
                fit_wall_s = _timed_end(device, fit_started)
                test_start = _timed_start(device)
                test_probabilities = _probabilities(frozen.modules, data.test, device)
                test_predict_time = _timed_end(device, test_start)
                validation_probabilities = _probabilities(frozen.modules, data.val, device)
                validation_metrics = _single_metrics(data.y_val, validation_probabilities)
                test_metrics = _single_metrics(data.y_test, test_probabilities)
                checkpoint_path = run_dir / "inference.pt"
                torch.save(_checkpoint_payload(method, seed, frozen), checkpoint_path)
                method_config = {"method": method, "config": config.to_dict()}
                result = {
                    "schema_version": 1,
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "attempt": 0,
                    "seed": seed,
                    "method": method,
                    "status": "complete",
                    "protocol_id": PROTOCOL_ID,
                    "comparison_hash": comparison_hash,
                    "method_config_hash": _sha256_json(method_config),
                    "resolved_config_sha256": _sha256_json({"config": config.to_dict(), "seed": seed, "method": method}),
                    "source": manifest["source"],
                    "data": data.dataset_id,
                    "environment_id": environment_id,
                    "environment_sha256": _sha256_json(environment),
                    "timing_protocol_id": "wall-v1-synchronized",
                    "metrics": {"validation": validation_metrics, "test": test_metrics},
                    "selection": frozen.selection,
                    "training": frozen.training,
                    "timing": {
                        "fit_wall_s": fit_wall_s,
                        "final_test_predict_wall_s": test_predict_time,
                        "run_wall_s": time.perf_counter() - run_started,
                    },
                    "artifacts": {"checkpoint": str(checkpoint_path.relative_to(root).as_posix())},
                    "environment": environment,
                    "error": None,
                }
                checkpoint_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
                result["artifacts"]["checkpoint_sha256"] = checkpoint_digest
                _write_json(run_dir / "result.json", result)
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)[:500]}
                result = {
                    "schema_version": 1, "experiment_id": experiment_id, "run_id": run_id,
                    "attempt": 0, "seed": seed, "method": method, "status": "failed",
                    "protocol_id": PROTOCOL_ID, "comparison_hash": comparison_hash,
                    "source": manifest["source"], "data": data.dataset_id,
                    "environment_id": environment_id, "environment_sha256": _sha256_json(environment),
                    "timing_protocol_id": "wall-v1-synchronized", "error": error,
                }
                _write_json(run_dir / "result.json", result)
                raise
    manifest["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(root / "manifest.json", manifest)
    return root


def smoke_comparison(output: str | Path, *, requested_device: str = "cpu") -> Path:
    """Run a tiny offline all-methods smoke test using the explicit fixture."""

    from .data import make_synthetic_fixture, prepare_data

    raw = make_synthetic_fixture()
    data = prepare_data(raw, n_quantiles=16, random_state=0)
    config = ReducedConfig(
        seeds=(0,), member_count=3, max_selected=3, max_epochs=3, patience=2, batch_size=8,
        heterogeneous_depths=(1, 2), heterogeneous_widths=(8, 12),
        heterogeneous_dropouts=(0.0, 0.1), heterogeneous_learning_rate_range=(0.001, 0.002),
        heterogeneous_weight_decay_range=(0.00001, 0.0001),
    )
    return run_comparison(
        data, config=config, output=output, requested_device=requested_device,
        experiment_id="synthetic-smoke-v1",
    )


def report_comparison(run_dir: str | Path, output: str | Path) -> Path:
    """Validate result artifacts and write strict JSON plus a readable Markdown report."""

    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for path in sorted(root.glob("*/result.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    manifest_hash = manifest.get("comparison_hash")
    if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
        raise ValueError("manifest comparison hash is invalid")
    for record in records:
        if record.get("comparison_hash") != manifest_hash:
            raise ValueError("result comparison hash does not match manifest")
        if record.get("status") != "complete":
            continue
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("checkpoint"), str):
            raise ValueError("complete result is missing its checkpoint artifact")
        checkpoint = (root / artifacts["checkpoint"]).resolve()
        if not checkpoint.is_relative_to(root.resolve()) or not checkpoint.is_file():
            raise ValueError("declared checkpoint artifact is missing or outside the run")
        expected_digest = artifacts.get("checkpoint_sha256")
        observed_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if expected_digest != observed_digest:
            raise ValueError("checkpoint artifact hash does not match result")
    summary = aggregate_results(
        records,
        expected_seeds=manifest["expected_seeds"],
        expected_methods=manifest["expected_methods"],
        strict=True,
    )
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "summary.json", summary)
    with (destination / "per_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "seed", "method", "validation_roc_auc", "validation_log_loss",
                "validation_accuracy", "test_roc_auc", "test_log_loss", "test_accuracy",
                "fit_wall_s", "final_test_predict_wall_s", "epochs_completed",
                "selected_snapshot_count", "result_path",
            ]
        )
        for result_path in sorted(root.glob("*/result.json")):
            record = json.loads(result_path.read_text(encoding="utf-8"))
            validation = record.get("metrics", {}).get("validation", {})
            test = record.get("metrics", {}).get("test", {})
            timing = record.get("timing", {})
            training = record.get("training", {})
            selection = record.get("selection", {})
            writer.writerow(
                [
                    record.get("seed"), record.get("method"), validation.get("roc_auc"),
                    validation.get("log_loss"), validation.get("accuracy"), test.get("roc_auc"),
                    test.get("log_loss"), test.get("accuracy"), timing.get("fit_wall_s"),
                    timing.get("final_test_predict_wall_s"), training.get("epochs_completed"),
                    selection.get("selected_snapshot_count"), result_path.relative_to(root).as_posix(),
                ]
            )
    lines = [
        "# Reduced Churn comparison",
        "",
        f"Protocol: `{manifest['protocol_id']}` (reduced optimistic workflow)",
        f"Comparison hash: `{manifest['comparison_hash']}`",
        f"Seeds: `{manifest['expected_seeds']}`; device: `{manifest['environment']['selected_device']}`",
        "",
        "Local test metrics are ROC-AUC/log-loss. The official paper's Churn table reports accuracy; these are not numerically comparable.",
        "",
        "| Method | Test ROC-AUC mean ± sample SD | Test log-loss mean ± sample SD | Fit seconds mean ± sample SD |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in manifest["expected_methods"]:
        group = summary["groups"][method]
        metrics = group["metrics"]["test"]
        timing = group["timing"]["fit_wall_s"]
        def fmt(item: Mapping[str, Any]) -> str:
            if item["mean"] is None:
                return "n/a"
            if item["sample_std"] is None:
                return f"{item['mean']:.6f}"
            return f"{item['mean']:.6f} ± {item['sample_std']:.6f}"
        lines.append(f"| `{method}` | {fmt(metrics['roc_auc'])} | {fmt(metrics['log_loss'])} | {fmt(timing)} |")
    lines += ["", "Every seed result is retained in `per_seed.csv` and under the local run directory; validation selection is frozen before test evaluation.", ""]
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return destination


__all__ = [
    "METHODS", "PROTOCOL_ID", "environment_record", "resolve_device", "run_comparison",
    "smoke_comparison", "report_comparison",
]
