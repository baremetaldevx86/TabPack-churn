"""Small, deterministic primitives for the reduced TabPack-style experiment.

The module deliberately has no NumPy, PyTorch, or scikit-learn dependency.  Model
code can pass detached one-dimensional predictions to the metric and selection
boundaries, while configuration sampling remains reproducible on every Python
platform.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import random
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

SEED_ALGORITHM = "sha256-named-v1"
SAMPLING_ALGORITHM = "sha256-named-v1/python-random-mt19937-v1"
_METHODS = {
    "homogeneous",
    "independent_mlp_ensemble",
    "heterogeneous",
    "packed_heterogeneous_ensemble",
}


def _is_int(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _finite_real(value: object, name: str) -> float:
    if not _is_real(value):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be representable as a finite float") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _as_1d(value: object, name: str) -> list[Any]:
    """Convert a normal sequence or an array/tensor-like value to a list.

    The conversion intentionally rejects two-dimensional inputs instead of
    flattening them: silently changing row order at a metric boundary is a
    particularly difficult experiment error to diagnose.
    """

    source = value
    if hasattr(source, "detach"):
        source = source.detach()
    if hasattr(source, "cpu"):
        source = source.cpu()
    ndim = getattr(source, "ndim", None)
    if ndim is not None and int(ndim) != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if hasattr(source, "tolist"):
        source = source.tolist()
    if isinstance(source, (str, bytes, Mapping, set, frozenset)):
        raise ValueError(f"{name} must be a one-dimensional sequence")
    try:
        result = list(source)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence") from exc
    if any(isinstance(item, (list, tuple)) for item in result):
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _validate_seed(seed: object, name: str = "seed") -> int:
    if not _is_int(seed):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(seed)
    if result < 0 or result >= 2**63:
        raise ValueError(f"{name} must be in [0, 2**63)")
    return result


def _canonical_part(value: object) -> str:
    """Type-tagged path encoding, avoiding process-randomized ``hash``."""

    if isinstance(value, bool):
        return "bool:" + ("1" if value else "0")
    if _is_int(value):
        return f"int:{int(value)}"
    if _is_real(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("seed path values must be finite")
        return "float:" + number.hex()
    if isinstance(value, str):
        return "str:" + value
    if value is None:
        return "none:"
    raise TypeError(f"unsupported named-seed path value: {type(value).__name__}")


def named_seed(root_seed: int, *path: object) -> int:
    """Derive a stable non-negative 63-bit seed from a named path.

    Every component is length-delimited and type-tagged.  Consequently this
    function is independent of Python's hash randomization, call order, and
    dictionary iteration order.  Named paths are domain-separated by the
    algorithm version and should include the purpose (for example,
    ``"member", 2, "initialization"``).
    """

    root = _validate_seed(root_seed, "root_seed")
    encoded = [_canonical_part(root), *(_canonical_part(part) for part in path)]
    payload = "tabpack-reduced|" + SEED_ALGORITHM + "|" + "|".join(
        f"{len(part)}:{part}" for part in encoded
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def derive_seed(root_seed: int, *path: object) -> int:
    """Alias for :func:`named_seed` used by execution-layer callers."""

    return named_seed(root_seed, *path)


def derive_named_seed(root_seed: int, *path: object) -> int:
    """Descriptive alias for :func:`named_seed`."""

    return named_seed(root_seed, *path)


def _validate_choice_list(value: object, name: str, *, integer: bool = False) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a non-empty sequence")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-empty sequence") from exc
    if not values:
        raise ValueError(f"{name} must not be empty")
    normalized: list[Any] = []
    for index, item in enumerate(values):
        if integer:
            if not _is_int(item) or int(item) <= 0:
                raise ValueError(f"{name}[{index}] must be a positive integer")
            item = int(item)
        else:
            item = _finite_real(item, f"{name}[{index}]")
        normalized.append(item)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate choices")
    return tuple(normalized)


def _validate_log_range(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must contain exactly two positive finite values")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly two positive finite values") from exc
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two positive finite values")
    low = _finite_real(values[0], f"{name}[0]")
    high = _finite_real(values[1], f"{name}[1]")
    if low <= 0 or high <= 0 or low > high:
        raise ValueError(f"{name} must satisfy 0 < low <= high")
    return (low, high)


@dataclass(frozen=True)
class ReducedConfig:
    """Validated defaults and search spaces for the reduced experiment."""

    seeds: tuple[int, ...] = (0, 1, 2)
    member_count: int = 4
    max_selected: int = 4
    max_epochs: int = 100
    patience: int = 16
    batch_size: int = 256
    dtype: str = "float32"
    baseline_depth: int = 2
    baseline_width: int = 64
    baseline_dropout: float = 0.1
    baseline_learning_rate: float = 0.001
    baseline_weight_decay: float = 0.0001
    heterogeneous_depths: tuple[int, ...] = (1, 2, 3)
    heterogeneous_widths: tuple[int, ...] = (32, 64, 96)
    heterogeneous_dropouts: tuple[float, ...] = (0.0, 0.1, 0.2)
    heterogeneous_learning_rate_range: tuple[float, float] = (0.0003, 0.003)
    heterogeneous_weight_decay_range: tuple[float, float] = (0.000001, 0.001)

    def __post_init__(self) -> None:
        # Normalize list inputs once, while retaining an immutable public record.
        if not isinstance(self.seeds, (tuple, list)):
            raise ValueError("seeds must be a non-empty ordered sequence")
        seeds = tuple(self.seeds)
        if not seeds:
            raise ValueError("seeds must not be empty")
        normalized_seeds = tuple(_validate_seed(seed, f"seeds[{i}]") for i, seed in enumerate(seeds))
        if len(set(normalized_seeds)) != len(normalized_seeds):
            raise ValueError("seeds must not contain duplicates")
        object.__setattr__(self, "seeds", normalized_seeds)

        for field in ("member_count", "max_selected", "max_epochs", "patience", "batch_size"):
            value = getattr(self, field)
            if not _is_int(value) or int(value) <= 0:
                raise ValueError(f"{field} must be a positive integer")
            object.__setattr__(self, field, int(value))
        if not isinstance(self.dtype, str) or self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")

        for field in ("baseline_depth", "baseline_width"):
            value = getattr(self, field)
            if not _is_int(value) or int(value) <= 0:
                raise ValueError(f"{field} must be a positive integer")
            object.__setattr__(self, field, int(value))
        for field in ("baseline_dropout", "baseline_learning_rate", "baseline_weight_decay"):
            value = _finite_real(getattr(self, field), field)
            object.__setattr__(self, field, value)
        if not 0 <= self.baseline_dropout < 1:
            raise ValueError("baseline_dropout must satisfy 0 <= value < 1")
        if self.baseline_learning_rate <= 0:
            raise ValueError("baseline_learning_rate must be positive")
        if self.baseline_weight_decay < 0:
            raise ValueError("baseline_weight_decay must be non-negative")

        object.__setattr__(
            self, "heterogeneous_depths", _validate_choice_list(self.heterogeneous_depths, "heterogeneous_depths", integer=True)
        )
        object.__setattr__(
            self, "heterogeneous_widths", _validate_choice_list(self.heterogeneous_widths, "heterogeneous_widths", integer=True)
        )
        dropouts = _validate_choice_list(self.heterogeneous_dropouts, "heterogeneous_dropouts")
        if any(not 0 <= value < 1 for value in dropouts):
            raise ValueError("heterogeneous_dropouts values must satisfy 0 <= value < 1")
        object.__setattr__(self, "heterogeneous_dropouts", dropouts)
        object.__setattr__(
            self,
            "heterogeneous_learning_rate_range",
            _validate_log_range(self.heterogeneous_learning_rate_range, "heterogeneous_learning_rate_range"),
        )
        object.__setattr__(
            self,
            "heterogeneous_weight_decay_range",
            _validate_log_range(self.heterogeneous_weight_decay_range, "heterogeneous_weight_decay_range"),
        )

    @property
    def n_members(self) -> int:
        return self.member_count

    @property
    def selection_limit(self) -> int:
        return self.max_selected

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ReducedConfig:
        """Parse a resolved nested config (or flat dataclass fields), rejecting typos."""

        if not isinstance(values, Mapping):
            raise ValueError("config must be a mapping")
        fields = dict(values)
        for section, names in (
            ("baseline", ("depth", "width", "dropout", "learning_rate", "weight_decay")),
            ("heterogeneous", ("depths", "widths", "dropouts", "learning_rate_range", "weight_decay_range")),
        ):
            if section not in fields:
                continue
            nested = fields.pop(section)
            if not isinstance(nested, Mapping):
                raise ValueError(f"{section} must be a mapping")
            for key, value in nested.items():
                if key not in names:
                    raise ValueError(f"unknown config field: {section}.{key}")
                field = f"{section}_{key}"
                if field in fields:
                    raise ValueError(f"duplicate config field: {field}")
                fields[field] = value
        unknown = set(fields) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown, key=str)}")
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe resolved configuration representation."""

        return {
            "seeds": list(self.seeds),
            "member_count": self.member_count,
            "max_selected": self.max_selected,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "dtype": self.dtype,
            "baseline": {
                "depth": self.baseline_depth,
                "width": self.baseline_width,
                "dropout": self.baseline_dropout,
                "learning_rate": self.baseline_learning_rate,
                "weight_decay": self.baseline_weight_decay,
            },
            "heterogeneous": {
                "depths": list(self.heterogeneous_depths),
                "widths": list(self.heterogeneous_widths),
                "dropouts": list(self.heterogeneous_dropouts),
                "learning_rate_range": list(self.heterogeneous_learning_rate_range),
                "weight_decay_range": list(self.heterogeneous_weight_decay_range),
            },
        }


@dataclass(frozen=True)
class MemberSpec:
    """One independently initialized member's resolved hyperparameters."""

    member_id: str
    member_index: int
    depth: int
    width: int
    dropout: float
    learning_rate: float
    weight_decay: float
    stream_seeds: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.member_id, str) or not self.member_id:
            raise ValueError("member_id must be a non-empty string")
        if not _is_int(self.member_index) or self.member_index < 0:
            raise ValueError("member_index must be a non-negative integer")
        object.__setattr__(self, "member_index", int(self.member_index))
        for field in ("depth", "width"):
            value = getattr(self, field)
            if not _is_int(value) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
            object.__setattr__(self, field, int(value))
        dropout = _finite_real(self.dropout, "dropout")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must satisfy 0 <= value < 1")
        learning_rate = _finite_real(self.learning_rate, "learning_rate")
        weight_decay = _finite_real(self.weight_decay, "weight_decay")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        object.__setattr__(self, "dropout", dropout)
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "weight_decay", weight_decay)
        try:
            streams = tuple((name, seed) for name, seed in self.stream_seeds)
        except (TypeError, ValueError) as exc:
            raise ValueError("stream_seeds must contain (name, seed) pairs") from exc
        if any(not isinstance(name, str) or not name for name, _ in streams):
            raise ValueError("stream_seeds names must be non-empty strings")
        if not streams or len({name for name, _ in streams}) != len(streams):
            raise ValueError("stream_seeds must contain unique named streams")
        object.__setattr__(
            self,
            "stream_seeds",
            tuple(sorted((name, _validate_seed(seed, f"stream_seeds[{name}]")) for name, seed in streams)),
        )

    @property
    def streams(self) -> dict[str, int]:
        return dict(self.stream_seeds)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> MemberSpec:
        """Rehydrate resolved member records without drawing new random values."""

        if not isinstance(values, Mapping) or set(values) != set(cls.__dataclass_fields__):
            raise ValueError("member record must contain exactly the MemberSpec fields")
        fields = dict(values)
        streams = fields.get("stream_seeds")
        if not isinstance(streams, Mapping):
            raise ValueError("stream_seeds must be a mapping")
        fields["stream_seeds"] = tuple(streams.items())
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "member_index": self.member_index,
            "depth": self.depth,
            "width": self.width,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "stream_seeds": dict(self.stream_seeds),
        }


def _log_uniform(rng: random.Random, bounds: tuple[float, float]) -> float:
    low, high = bounds
    if low == high:
        return low
    # Clamp only round-off at the endpoints, not the distribution itself.
    return min(high, max(low, math.exp(math.log(low) + rng.random() * (math.log(high) - math.log(low)))))


def sample_members(
    config: ReducedConfig,
    seed: int,
    *,
    mode: str = "heterogeneous",
    heterogeneous: bool | None = None,
) -> tuple[MemberSpec, ...]:
    """Resolve independent member records using named, local RNG streams.

    ``mode='homogeneous'`` creates baseline members with distinct stream IDs;
    heterogeneous mode draws each architecture and optimizer value from the
    validated spaces.  No global RNG is read or advanced.
    """

    if not isinstance(config, ReducedConfig):
        raise TypeError("config must be a ReducedConfig")
    root = _validate_seed(seed)
    if heterogeneous is not None:
        if not isinstance(heterogeneous, bool):
            raise ValueError("heterogeneous must be a boolean")
        mode = "heterogeneous" if heterogeneous else "homogeneous"
    if not isinstance(mode, str) or mode not in _METHODS:
        raise ValueError(f"unsupported member sampling mode: {mode!r}")
    is_homogeneous = mode in {"homogeneous", "independent_mlp_ensemble"}
    mode = "homogeneous" if is_homogeneous else "heterogeneous"
    members: list[MemberSpec] = []
    for index in range(config.member_count):
        member_id = f"member-{index:04d}"
        base_path = ("sampling", mode, "member", index)
        if is_homogeneous:
            depth = config.baseline_depth
            width = config.baseline_width
            dropout = config.baseline_dropout
            learning_rate = config.baseline_learning_rate
            weight_decay = config.baseline_weight_decay
        else:
            depth = config.heterogeneous_depths[
                random.Random(named_seed(root, *base_path, "depth")).randrange(len(config.heterogeneous_depths))
            ]
            width = config.heterogeneous_widths[
                random.Random(named_seed(root, *base_path, "width")).randrange(len(config.heterogeneous_widths))
            ]
            dropout = config.heterogeneous_dropouts[
                random.Random(named_seed(root, *base_path, "dropout")).randrange(len(config.heterogeneous_dropouts))
            ]
            learning_rate = _log_uniform(
                random.Random(named_seed(root, *base_path, "learning_rate")),
                config.heterogeneous_learning_rate_range,
            )
            weight_decay = _log_uniform(
                random.Random(named_seed(root, *base_path, "weight_decay")),
                config.heterogeneous_weight_decay_range,
            )
        streams = tuple(
            (purpose, named_seed(root, "training", mode, "member", index, purpose))
            for purpose in ("initialization", "permutation", "dropout")
        )
        members.append(
            MemberSpec(
                member_id=member_id,
                member_index=index,
                depth=depth,
                width=width,
                dropout=dropout,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                stream_seeds=streams,
            )
        )
    return tuple(members)


def sample_member_specs(
    config: ReducedConfig, seed: int, *, method: str = "heterogeneous"
) -> tuple[MemberSpec, ...]:
    """Descriptive alias for :func:`sample_members`."""

    return sample_members(config, seed, mode=method)


def average_probabilities(predictions: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Average positive-class probabilities row-wise using ``math.fsum``."""

    if isinstance(predictions, (str, bytes)):
        raise ValueError("predictions must be a non-empty sequence of vectors")
    try:
        vectors = [_as_1d(vector, "probability vector") for vector in predictions]
    except TypeError as exc:
        raise ValueError("predictions must be a non-empty sequence of vectors") from exc
    if not vectors:
        raise ValueError("predictions must not be empty")
    length = len(vectors[0])
    if length == 0:
        raise ValueError("probability vectors must not be empty")
    normalized: list[list[float]] = []
    for row_index, vector in enumerate(vectors):
        if len(vector) != length:
            raise ValueError("probability vectors must have equal lengths")
        row: list[float] = []
        for col_index, value in enumerate(vector):
            number = _finite_real(value, f"predictions[{row_index}][{col_index}]")
            if not 0 <= number <= 1:
                raise ValueError("probabilities must be in [0, 1]")
            row.append(number)
        normalized.append(row)
    count = len(normalized)
    return tuple(math.fsum(normalized[row][column] for row in range(count)) / count for column in range(length))


def _validate_binary_inputs(y_true: object, values: object, value_name: str) -> tuple[list[int], list[float]]:
    labels_raw = _as_1d(y_true, "y_true")
    values_raw = _as_1d(values, value_name)
    if not labels_raw or len(labels_raw) != len(values_raw):
        raise ValueError("y_true and prediction values must be non-empty and have equal lengths")
    labels: list[int] = []
    for index, value in enumerate(labels_raw):
        if not isinstance(value, numbers.Real) or not math.isfinite(float(value)) or float(value) not in (0.0, 1.0):
            raise ValueError(f"y_true[{index}] must be binary 0 or 1")
        labels.append(int(float(value)))
    converted: list[float] = []
    for index, value in enumerate(values_raw):
        converted.append(_finite_real(value, f"{value_name}[{index}]"))
    return labels, converted


def binary_roc_auc(y_true: object, y_score: object) -> float:
    """Return tie-aware binary ROC-AUC, awarding half credit for exact ties."""

    labels, scores = _validate_binary_inputs(y_true, y_score, "y_score")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC requires both classes")
    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    negatives_before = 0
    wins = 0.0
    position = 0
    while position < len(ranked):
        end = position + 1
        score = ranked[position][0]
        while end < len(ranked) and ranked[end][0] == score:
            end += 1
        group_positives = sum(label for _, label in ranked[position:end])
        group_negatives = (end - position) - group_positives
        wins += group_positives * negatives_before + 0.5 * group_positives * group_negatives
        negatives_before += group_negatives
        position = end
    return float(wins / (positives * negatives))


def binary_log_loss(y_true: object, y_prob: object, *, eps: float | None = None) -> float:
    """Return mean binary cross-entropy after strict probability validation."""

    labels, probabilities = _validate_binary_inputs(y_true, y_prob, "y_prob")
    if any(value < 0 or value > 1 for value in probabilities):
        raise ValueError("y_prob values must be probabilities in [0, 1]")
    epsilon = sys.float_info.epsilon if eps is None else _finite_real(eps, "eps")
    if not 0 < epsilon < 0.5 or 1.0 - epsilon >= 1.0:
        raise ValueError("eps must satisfy 0 < eps < 0.5 and be representable")
    losses = []
    for label, probability in zip(labels, probabilities):
        clipped = min(max(probability, epsilon), 1.0 - epsilon)
        losses.append(-math.log(clipped) if label else -math.log1p(-clipped))
    return float(math.fsum(losses) / len(losses))


@dataclass(frozen=True)
class Candidate:
    """An immutable validation prediction snapshot used by greedy selection."""

    candidate_id: str
    member_id: str
    source_epoch: int
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(self.member_id, str) or not self.member_id:
            raise ValueError("member_id must be a non-empty string")
        if not _is_int(self.source_epoch) or self.source_epoch < 0:
            raise ValueError("source_epoch must be a non-negative integer")
        values = _as_1d(self.probabilities, "probabilities")
        if not values:
            raise ValueError("probabilities must not be empty")
        normalized = []
        for index, value in enumerate(values):
            number = _finite_real(value, f"probabilities[{index}]")
            if not 0 <= number <= 1:
                raise ValueError("probabilities must be in [0, 1]")
            normalized.append(number)
        object.__setattr__(self, "probabilities", tuple(normalized))

    @property
    def canonical_key(self) -> tuple[str, str, int]:
        return (self.candidate_id, self.member_id, int(self.source_epoch))


@dataclass(frozen=True)
class SelectionStep:
    round_index: int
    candidate_scores: tuple[tuple[str, float], ...]
    winner_id: str | None
    accepted: bool
    resulting_auc: float


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[Candidate, ...]
    validation_auc: float
    candidate_order: tuple[str, ...]
    steps: tuple[SelectionStep, ...]
    validation_evaluations: int
    stop_reason: str

    @property
    def selected_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.selected)

    @property
    def selected_indices(self) -> tuple[int, ...]:
        """Original pool indices, in greedy acceptance order."""

        return tuple(self.candidate_order.index(candidate_id) for candidate_id in self.selected_ids)

    @property
    def probabilities(self) -> tuple[float, ...]:
        return average_probabilities([candidate.probabilities for candidate in self.selected])


def _validate_candidates(candidates: Sequence[Candidate], expected_length: int | None = None) -> list[Candidate]:
    try:
        values = list(candidates)
    except TypeError as exc:
        raise ValueError("candidates must be a sequence") from exc
    if not values:
        raise ValueError("candidates must not be empty: a best singleton is required")
    if any(not isinstance(candidate, Candidate) for candidate in values):
        raise ValueError("candidates must contain Candidate values")
    ids = [candidate.candidate_id for candidate in values]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_id values must be unique")
    if expected_length is not None and any(len(candidate.probabilities) != expected_length for candidate in values):
        raise ValueError("all candidate probabilities must match y_validation length")
    return values


def greedy_select(
    y_validation: object,
    candidates: Sequence[Candidate],
    *,
    max_size: int | None = None,
    max_ensemble_size: int | None = None,
) -> SelectionResult:
    """Select a probability-average ensemble by strict validation-AUC gains.

    The first member is always the best singleton.  Later rounds evaluate every
    unused snapshot, choose prospective AUC then individual AUC then original
    pool order, and stop on an exact non-improvement. Candidate identity, rather
    than ``member_id``, defines no-replacement semantics. Rebuild from scratch
    each epoch with previous selected snapshots followed by current snapshots.
    """

    labels = _as_1d(y_validation, "y_validation")
    # Use the metric's strict label validation even before inspecting candidates.
    _validate_binary_inputs(labels, [0.0] * len(labels), "validation")
    for name, value in (("max_size", max_size), ("max_ensemble_size", max_ensemble_size)):
        if value is not None and (not _is_int(value) or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    if max_size is not None and max_ensemble_size is not None and max_size != max_ensemble_size:
        raise ValueError("max_size and max_ensemble_size disagree")
    limit = max_size if max_size is not None else max_ensemble_size
    if limit is None:
        raise ValueError("max_size is required")
    if not _is_int(limit) or int(limit) <= 0:
        raise ValueError("max_size must be a positive integer")
    limit = int(limit)
    pool = _validate_candidates(candidates, len(labels))
    candidate_order = tuple(candidate.candidate_id for candidate in pool)
    individual_scores = {
        candidate.candidate_id: binary_roc_auc(labels, candidate.probabilities) for candidate in pool
    }
    # max() retains the first item on exact ties, preserving the input pool order.
    best = max(pool, key=lambda candidate: individual_scores[candidate.candidate_id])
    selected = [best]
    current_auc = individual_scores[best.candidate_id]
    ordered_remaining = [candidate for candidate in pool if candidate is not best]
    steps = [SelectionStep(0, tuple(individual_scores.items()), best.candidate_id, True, current_auc)]
    evaluations = len(individual_scores)
    round_index = 1
    while len(selected) < limit and ordered_remaining:
        scored: list[tuple[Candidate, float]] = []
        for candidate in ordered_remaining:
            aggregate = average_probabilities(
                [member.probabilities for member in [*selected, candidate]]
            )
            score = binary_roc_auc(labels, aggregate)
            evaluations += 1
            scored.append((candidate, score))
        winner, best_score = max(
            scored,
            key=lambda item: (item[1], individual_scores[item[0].candidate_id]),
        )
        score_by_id = tuple((candidate.candidate_id, score) for candidate, score in scored)
        if best_score <= current_auc:
            steps.append(SelectionStep(round_index, score_by_id, winner.candidate_id, False, current_auc))
            return SelectionResult(tuple(selected), current_auc, candidate_order, tuple(steps), evaluations, "no_strict_improvement")
        selected.append(winner)
        ordered_remaining = [candidate for candidate in ordered_remaining if candidate.candidate_id != winner.candidate_id]
        current_auc = best_score
        steps.append(SelectionStep(round_index, score_by_id, winner.candidate_id, True, current_auc))
        round_index += 1
    reason = "max_size" if len(selected) >= limit else "pool_exhausted"
    return SelectionResult(tuple(selected), current_auc, candidate_order, tuple(steps), evaluations, reason)


def select_greedy(
    y_validation: object,
    candidates: Sequence[Candidate],
    *,
    max_size: int | None = None,
    max_ensemble_size: int | None = None,
) -> SelectionResult:
    """Compatibility alias for :func:`greedy_select`."""

    return greedy_select(
        y_validation,
        candidates,
        max_size=max_size,
        max_ensemble_size=max_ensemble_size,
    )


def retain_best_selection(
    incumbent: SelectionResult | None, proposed: SelectionResult
) -> SelectionResult:
    """Accept a rebuilt online ensemble only on strict validation improvement."""

    if incumbent is None or proposed.validation_auc > incumbent.validation_auc:
        return proposed
    return incumbent


def mean_sample_std(values: Iterable[float]) -> tuple[float | None, float | None]:
    """Return a finite arithmetic mean and sample standard deviation."""

    numbers_list = [_finite_real(value, "values") for value in values]
    if not numbers_list:
        return (None, None)
    # statistics uses exact rational accumulation for finite binary64 inputs;
    # this also handles large near-identical times without raw-moment cancellation.
    mean = float(statistics.mean(numbers_list))
    if len(numbers_list) < 2:
        return (mean, None)
    std = statistics.stdev(numbers_list)
    if not math.isfinite(std):
        raise ValueError("sample standard deviation is not representable as a finite float")
    return (mean, std)


def sample_mean_std(values: Iterable[float]) -> dict[str, float | int | None]:
    """Mapping form of :func:`mean_sample_std` for JSON/reporting callers."""

    values_list = list(values)
    mean, std = mean_sample_std(values_list)
    return {"n": len(values_list), "mean": mean, "sample_std": std}


def _record_metric(record: Mapping[str, Any], split: str, metric: str) -> object:
    nested = record.get("metrics")
    if "metrics" in record and not isinstance(nested, Mapping):
        raise ValueError("metrics must be a mapping")
    if isinstance(nested, Mapping):
        split_values = nested.get(split)
        if split in nested and not isinstance(split_values, Mapping):
            raise ValueError(f"metrics.{split} must be a mapping")
        if isinstance(split_values, Mapping) and metric in split_values:
            return split_values[metric]
    for key in (f"{split}_{metric}", metric if split == "test" else ""):
        if key and key in record:
            return record[key]
    return None


def aggregate_results(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int] | None = None,
    expected_methods: Sequence[str] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Aggregate complete run records by method, seed, split, and metric.

    Failed/cancelled records remain visible in diagnostics and never contribute
    values. All complete records require validation/test ROC-AUC and log-loss.
    The returned object is JSON-safe and deterministically ordered. The default
    raises when the declared seed/method matrix is incomplete. ``strict=False``
    permits a visibly incomplete diagnostic summary, never malformed metrics or
    duplicate/undeclared runs. This is an in-memory reduction helper, not an
    artifact/hash validator; the caller validates files against its manifest.
    """

    materialized = []
    if not isinstance(strict, bool):
        raise ValueError("strict must be a boolean")
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("result records must be mappings")
        materialized.append(dict(record))
    declared_seeds = None
    if expected_seeds is not None:
        declared_seeds = sorted(_validate_seed(seed) for seed in expected_seeds)
        if not declared_seeds or len(set(declared_seeds)) != len(declared_seeds):
            raise ValueError("expected_seeds must be non-empty and unique")
    methods = None if expected_methods is None else list(expected_methods)
    if isinstance(expected_methods, (str, bytes)):
        raise ValueError("expected_methods must be a sequence of method strings")
    if methods is not None:
        if not methods or any(not isinstance(method, str) or not method for method in methods):
            raise ValueError("expected_methods must contain non-empty method strings")
        if len(set(methods)) != len(methods):
            raise ValueError("expected_methods must be unique")
    if not materialized and (declared_seeds is None or methods is None):
        raise ValueError("empty results require explicit expected_seeds and expected_methods")
    diagnostics: dict[str, Any] = {"failed": [], "cancelled": [], "missing": []}
    complete_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    observed_keys: set[tuple[str, int]] = set()
    run_ids: set[str] = set()
    common_fields = (
        "experiment_id", "comparison_hash", "protocol_id", "source", "data",
        "environment_id", "environment_sha256", "timing_protocol_id",
    )
    # These identities must be identical within one reduction. A caller that
    # wants several contexts can invoke this helper separately for each context.
    context: dict[str, object] = {}
    for field in common_fields:
        if any(field in record for record in materialized):
            if any(field not in record for record in materialized):
                raise ValueError(f"inconsistent result context: missing {field}")
            value = materialized[0][field]
            if any(record[field] != value for record in materialized):
                raise ValueError(f"incompatible result context: {field}")
            context[field] = value
    for field in ("method_config_hash",):
        for method in {record.get("method") for record in materialized if isinstance(record.get("method"), str)}:
            method_records = [record for record in materialized if record.get("method") == method]
            if any(field in record for record in method_records):
                if any(field not in record for record in method_records) or any(
                    record[field] != method_records[0][field] for record in method_records
                ):
                    raise ValueError(f"incompatible {method} result context: {field}")
    for record in materialized:
        method = record.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError("each result record requires a non-empty method")
        seed = _validate_seed(record.get("seed"), "record.seed")
        if "schema_version" in record:
            version = record["schema_version"]
            if not _is_int(version) or version != 1:
                raise ValueError("unsupported result schema_version")
        if methods is not None and method not in methods:
            raise ValueError(f"unexpected method: {method}")
        if declared_seeds is not None and seed not in declared_seeds:
            raise ValueError(f"unexpected seed: {seed}")
        status = record.get("status")
        if status not in ("complete", "failed", "cancelled"):
            raise ValueError(f"invalid result status for {method}/{seed}: {status!r}")
        key = (method, seed)
        observed_keys.add(key)
        run_id = record.get("run_id")
        if run_id is not None:
            if not isinstance(run_id, str) or not run_id or run_id in run_ids:
                raise ValueError("run_id values must be non-empty and unique")
            run_ids.add(run_id)
        if status in {"failed", "cancelled"}:
            diagnostics[status].append({"method": method, "seed": seed})
            continue
        if key in complete_by_key:
            raise ValueError(f"duplicate complete runs for {method}/{seed}; resolve retries explicitly")
        for split in ("validation", "test"):
            for metric in ("roc_auc", "log_loss"):
                value = _record_metric(record, split, metric)
                number = _finite_real(value, f"{method}/{seed}/{split}/{metric}")
                if number < 0 or (metric == "roc_auc" and number > 1):
                    raise ValueError(f"invalid {method}/{seed}/{split}/{metric}")
        complete_by_key[key] = record

    if methods is None:
        methods = sorted({method for method, _ in observed_keys})
    if declared_seeds is None:
        declared_seeds = sorted({seed for _, seed in observed_keys})

    groups: dict[str, Any] = {}
    for method in methods:
        observed = sorted(seed for (record_method, seed) in complete_by_key if record_method == method)
        expected = declared_seeds
        missing = [seed for seed in expected if (method, seed) not in observed_keys]
        incomplete = [seed for seed in expected if seed not in observed]
        diagnostics["missing"].extend({"method": method, "seed": seed} for seed in missing)
        group: dict[str, Any] = {
            "method": method,
            "expected_seeds": list(expected),
            "seeds": observed,
            "missing_seeds": missing,
            "incomplete_seeds": incomplete,
            "failed_seeds": sorted({item["seed"] for item in diagnostics["failed"] if item["method"] == method}),
            "cancelled_seeds": sorted({item["seed"] for item in diagnostics["cancelled"] if item["method"] == method}),
            "n_expected": len(expected),
            "n_complete": len(observed),
            "complete": not incomplete,
            "metrics": {},
        }
        for split in ("validation", "test"):
            split_summary: dict[str, Any] = {}
            for metric in ("roc_auc", "log_loss"):
                values = []
                value_seeds = []
                for seed in observed:
                    value = _record_metric(complete_by_key[(method, seed)], split, metric)
                    values.append(_finite_real(value, f"{method}/{seed}/{split}/{metric}"))
                    value_seeds.append(seed)
                summary = sample_mean_std(values)
                split_summary[metric] = {
                    **summary,
                    "n_observed": len(values),
                    "seeds": value_seeds,
                }
            group["metrics"][split] = split_summary
        group["timing"] = {}
        for field in ("fit_wall_s", "final_test_predict_wall_s", "run_wall_s"):
            values = []
            value_seeds = []
            for seed in observed:
                record = complete_by_key[(method, seed)]
                timing = record.get("timing", {})
                if not isinstance(timing, Mapping):
                    raise ValueError(f"{method}/{seed}/timing must be a mapping")
                value = timing.get(field, record.get(field))
                if value is None:
                    continue
                number = _finite_real(value, f"{method}/{seed}/timing/{field}")
                if number < 0:
                    raise ValueError(f"{method}/{seed}/timing/{field} must be non-negative")
                values.append(number)
                value_seeds.append(seed)
            group["timing"][field] = {
                **sample_mean_std(values), "n_observed": len(values), "seeds": value_seeds,
            }
        groups[method] = group
    for key in diagnostics:
        diagnostics[key].sort(key=lambda item: (item.get("method", ""), item.get("seed", -1)))
    result = {
        "complete": all(group["complete"] for group in groups.values()),
        "context": context, "groups": groups, "diagnostics": diagnostics,
    }
    # Context metadata can be supplied by a manifest; refuse non-finite or
    # otherwise non-JSON-safe objects rather than advertising a safe summary.
    canonical_json(context)
    if strict and not result["complete"]:
        raise ValueError("result matrix is incomplete")
    return result


def summarize_results(
    records: Iterable[Mapping[str, Any]], **kwargs: Any
) -> dict[str, Any]:
    """Alias for :func:`aggregate_results`."""

    return aggregate_results(records, **kwargs)


def aggregate_runs(records: Iterable[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Alias for :func:`aggregate_results`."""

    return aggregate_results(records, **kwargs)


def paired_deltas(
    baseline: Mapping[int, float], candidate: Mapping[int, float]
) -> dict[str, Any]:
    """Join by seed and summarize candidate-minus-baseline deltas."""

    left = {_validate_seed(seed): _finite_real(value, "baseline") for seed, value in baseline.items()}
    right = {_validate_seed(seed): _finite_real(value, "candidate") for seed, value in candidate.items()}
    seeds = sorted(left.keys() & right.keys())
    deltas = [right[seed] - left[seed] for seed in seeds]
    return {
        **sample_mean_std(deltas), "seeds": seeds, "deltas": deltas,
        "excluded_baseline_seeds": sorted(left.keys() - right.keys()),
        "excluded_candidate_seeds": sorted(right.keys() - left.keys()),
    }


def canonical_json(value: object) -> str:
    """Deterministic strict JSON suitable for hashing resolved core records."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = [
    "Candidate",
    "MemberSpec",
    "ReducedConfig",
    "SAMPLING_ALGORITHM",
    "SEED_ALGORITHM",
    "SelectionResult",
    "SelectionStep",
    "aggregate_results",
    "aggregate_runs",
    "average_probabilities",
    "binary_log_loss",
    "binary_roc_auc",
    "canonical_json",
    "derive_named_seed",
    "derive_seed",
    "greedy_select",
    "mean_sample_std",
    "named_seed",
    "paired_deltas",
    "retain_best_selection",
    "sample_mean_std",
    "sample_member_specs",
    "sample_members",
    "select_greedy",
    "summarize_results",
]
