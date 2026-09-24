"""Independent tiny oracles for the dependency-light core primitives."""

import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pytest

from tabpack.core import (
    Candidate,
    MemberSpec,
    ReducedConfig,
    aggregate_results,
    average_probabilities,
    binary_log_loss,
    binary_roc_auc,
    canonical_json,
    greedy_select,
    mean_sample_std,
    named_seed,
    paired_deltas,
    retain_best_selection,
    sample_members,
)


def pairwise_auc(labels, scores):
    positive = [score for label, score in zip(labels, scores) if label == 1]
    negative = [score for label, score in zip(labels, scores) if label == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def test_config_defaults_round_trip_and_immutability():
    config = ReducedConfig()
    assert config.seeds == (0, 1, 2)
    assert (config.member_count, config.max_selected, config.max_epochs) == (4, 4, 100)
    assert (config.patience, config.batch_size, config.dtype) == (16, 256, "float32")
    assert ReducedConfig.from_dict(json.loads(canonical_json(config.to_dict()))) == config
    seeds = [0, 1]
    choices = [1, 2]
    copied = ReducedConfig(seeds=seeds, heterogeneous_depths=choices)
    seeds.append(3)
    choices.append(4)
    assert copied.seeds == (0, 1)
    assert copied.heterogeneous_depths == (1, 2)
    with pytest.raises(FrozenInstanceError):
        config.batch_size = 4
    # Old selected snapshots can make the candidate pool larger than member_count.
    assert ReducedConfig(member_count=1, max_selected=4, patience=200).max_selected == 4


@pytest.mark.parametrize("field", ["member_count", "max_selected", "max_epochs", "patience", "batch_size", "baseline_depth", "baseline_width"])
@pytest.mark.parametrize("invalid", [0, -1, 1.5, True, None, "4"])
def test_positive_integral_config_fields(field, invalid):
    with pytest.raises(ValueError, match=field):
        ReducedConfig(**{field: invalid})


@pytest.mark.parametrize("seeds", [[], None, "01", [0, 0], [-1], [True], [0.5], [2**63]])
def test_config_seed_validation(seeds):
    with pytest.raises(ValueError, match="seeds"):
        ReducedConfig(seeds=seeds)


@pytest.mark.parametrize("field,invalid", [
    ("baseline_dropout", 1), ("baseline_dropout", -0.1),
    ("baseline_dropout", float("nan")), ("baseline_learning_rate", 0),
    ("baseline_learning_rate", "0.1"), ("baseline_learning_rate", True),
    ("baseline_weight_decay", -0.1), ("baseline_weight_decay", float("inf")),
    ("heterogeneous_depths", []), ("heterogeneous_depths", [1, 1]),
    ("heterogeneous_depths", [1, True]), ("heterogeneous_widths", {32, 64}),
    ("heterogeneous_widths", [0]), ("heterogeneous_dropouts", [0.1, 0.1]),
    ("heterogeneous_dropouts", [float("nan")]), ("heterogeneous_dropouts", [1]),
    ("dtype", "float16"),
])
def test_invalid_config_values(field, invalid):
    with pytest.raises(ValueError, match=field):
        ReducedConfig(**{field: invalid})


@pytest.mark.parametrize("field", ["heterogeneous_learning_rate_range", "heterogeneous_weight_decay_range"])
@pytest.mark.parametrize("invalid", [(0, 1), (-1, 1), (1, 0.1), (1,), (1, 2, 3), "12", None, (True, 2), (1, float("inf")), (float("nan"), 1)])
def test_invalid_log_ranges(field, invalid):
    with pytest.raises(ValueError, match=field):
        ReducedConfig(**{field: invalid})


def test_unknown_and_malformed_config_keys_fail():
    for values in ({"learnng_rate": 1}, {"baseline": {"wd": 1}}, {"baseline": []},
                   {"baseline": {"width": 1}, "baseline_width": 2}):
        with pytest.raises(ValueError):
            ReducedConfig.from_dict(values)


def test_named_seed_independent_digest_oracle_and_path_separation():
    # Exact encoding is documented and independently constructed here.
    payload = b"tabpack-reduced|sha256-named-v1|5:int:0|18:str:initialization|5:int:2"
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
    assert named_seed(0, "initialization", 2) == expected
    assert named_seed(0, "a|b", "c") != named_seed(0, "a", "b|c")
    assert named_seed(0, 1) != named_seed(0, "1")
    assert named_seed(0, 1) != named_seed(0, True)
    assert named_seed(0, "init", 1) != named_seed(0, "dropout", 1)
    for invalid in (True, -1, 0.2, "1", 2**63):
        with pytest.raises(ValueError):
            named_seed(invalid)
    assert 0 <= named_seed(2**63 - 1, "boundary") < 2**63


def test_sampling_reproducibility_rng_isolation_and_prefix_stability():
    config = ReducedConfig()
    state = random.getstate()
    first = sample_members(config, 7)
    assert random.getstate() == state
    random.Random(99).random()
    assert first == sample_members(config, 7)
    assert first[:1] == sample_members(replace(config, member_count=1), 7)
    assert first != sample_members(config, 8)
    assert len({member.member_id for member in first}) == 4
    assert len({seed for member in first for _, seed in member.stream_seeds}) == 12
    for member in first:
        assert member.depth in config.heterogeneous_depths
        assert member.width in config.heterogeneous_widths
        assert member.dropout in config.heterogeneous_dropouts
        assert 0.0003 <= member.learning_rate <= 0.003
        assert 0.000001 <= member.weight_decay <= 0.001
    # Altering one hyperparameter's choices does not advance another field stream.
    changed = sample_members(replace(config, heterogeneous_depths=(1,)), 7)
    assert [member.width for member in changed] == [member.width for member in first]
    assert [member.stream_seeds for member in changed] == [member.stream_seeds for member in first]


def test_homogeneous_fixed_ranges_and_alias_modes():
    config = ReducedConfig(baseline_weight_decay=0)
    members = sample_members(config, 0, mode="homogeneous")
    assert members == sample_members(config, 0, mode="independent_mlp_ensemble")
    for member in members:
        assert (member.depth, member.width, member.dropout) == (2, 64, 0.1)
        assert (member.learning_rate, member.weight_decay) == (0.001, 0)
    fixed = ReducedConfig(
        heterogeneous_depths=[1], heterogeneous_widths=[2], heterogeneous_dropouts=[0],
        heterogeneous_learning_rate_range=[0.002, 0.002],
        heterogeneous_weight_decay_range=[1e-5, 1e-5],
    )
    for member in sample_members(fixed, 9):
        assert (member.depth, member.width, member.dropout) == (1, 2, 0)
        assert member.learning_rate == 0.002
        assert member.weight_decay == 1e-5


def test_member_record_json_round_trip_without_resampling():
    member = sample_members(ReducedConfig(), 0)[0]
    decoded = json.loads(canonical_json(member.to_dict()))
    restored = MemberSpec.from_dict(decoded)
    assert restored == member
    assert restored.to_dict() == member.to_dict()
    streams = restored.streams
    streams["dropout"] = 0
    assert restored.streams["dropout"] == member.streams["dropout"]
    with pytest.raises(ValueError):
        MemberSpec.from_dict({**decoded, "unknown": 1})


def test_log_uniform_draw_independent_oracle():
    config = ReducedConfig(member_count=1)
    # Independent derivation of the field RNG; avoid the production seed helper.
    parts = ["int:11", "str:sampling", "str:heterogeneous", "str:member", "int:0", "str:learning_rate"]
    encoded = "tabpack-reduced|sha256-named-v1|" + "|".join(f"{len(part)}:{part}" for part in parts)
    seed = int.from_bytes(hashlib.sha256(encoded.encode()).digest()[:8], "big") % (2**63)
    u = random.Random(seed).random()
    expected = 0.0003 * (0.003 / 0.0003) ** u
    assert sample_members(config, 11)[0].learning_rate == pytest.approx(expected, rel=1e-14)


def test_sampling_ignores_python_hash_seed():
    code = "from tabpack.core import *; print(canonical_json([m.to_dict() for m in sample_members(ReducedConfig(), 7)]))"
    outputs = [subprocess.check_output(
        [sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": str(seed)}, text=True,
    ) for seed in (1, 19)]
    assert outputs[0] == outputs[1]


def test_auc_exhaustive_small_pairwise_oracle():
    count = 0
    for size in range(2, 6):
        for labels in itertools.product((0, 1), repeat=size):
            if len(set(labels)) == 1:
                continue
            for scores in itertools.product((0.0, 0.5, 1.0), repeat=size):
                assert binary_roc_auc(labels, scores) == pairwise_auc(labels, scores)
                count += 1
    assert count == 8604


@pytest.mark.parametrize("labels,scores,expected", [
    ([0, 1], [0.1, 0.9], 1), ([0, 1], [0.9, 0.1], 0),
    ([0, 1], [0.5, 0.5], 0.5), ([0, 1, 0, 1], [0.1, 0.4, 0.4, 0.8], 0.875),
    ([False, True], [-9, 12], 1),
])
def test_auc_named_ties_and_unbounded_scores(labels, scores, expected):
    assert binary_roc_auc(labels, scores) == expected


@pytest.mark.parametrize("labels,scores", [
    ([], []), ([0, 1], [0.1]), ([0, 2], [0.1, 0.9]), ([0, "1"], [0.1, 0.9]),
    ([0, 1], [0.1, float("nan")]), ([0, 1], [0.1, float("inf")]),
    ([[0], [1]], [0.1, 0.9]), ([0, 1], [[0.1], [0.9]]), ([0, 1], [True, False]),
])
def test_metrics_reject_malformed_inputs(labels, scores):
    for metric in (binary_roc_auc, binary_log_loss):
        with pytest.raises(ValueError):
            metric(labels, scores)


def test_auc_requires_both_classes_but_log_loss_does_not():
    for labels in ([0, 0], [1, 1]):
        with pytest.raises(ValueError, match="both classes"):
            binary_roc_auc(labels, [0.2, 0.8])
    assert binary_log_loss([1, 1], [0.8, 0.6]) == pytest.approx(0.3669845875401002)


def test_log_loss_endpoints_and_epsilon():
    assert binary_log_loss([0, 1], [0.1, 0.9]) == pytest.approx(-math.log(0.9))
    assert binary_log_loss([0, 1], [0.5, 0.5]) == pytest.approx(math.log(2))
    assert binary_log_loss([0, 1], [0, 1]) == pytest.approx(-math.log1p(-2**-52), rel=1e-15, abs=0)
    assert binary_log_loss([0, 1], [1, 0]) == pytest.approx(-math.log(2**-52))
    assert binary_log_loss([0, 1], [1, 0], eps=0.125) == pytest.approx(-math.log(0.125))
    for invalid in (0, -1, 0.5, 1e-300, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="eps"):
            binary_log_loss([0, 1], [0, 1], eps=invalid)
    for probabilities in ([-0.1, 0.2], [0.1, 1.01]):
        with pytest.raises(ValueError):
            binary_log_loss([0, 1], probabilities)


def test_probability_averaging_uses_values_not_logits_or_member_scores():
    predictions = [[0.01, 0.6], [0.8, 0.1]]
    assert average_probabilities(predictions) == pytest.approx((0.405, 0.35))
    assert predictions == [[0.01, 0.6], [0.8, 0.1]]
    for invalid in ([], [[]], [[0.1], [0.1, 0.2]], [0.1, 0.2], [[-0.1]], [[float("nan")]]):
        with pytest.raises(ValueError):
            average_probabilities(invalid)


LABELS = (0, 0, 0, 1, 1, 1)


def candidates_from_grid(rows):
    return tuple(Candidate(f"candidate-{i}", "same-member", i, tuple(v / 16 for v in row)) for i, row in enumerate(rows))


def test_greedy_best_singleton_and_collective_improvements():
    pool = candidates_from_grid([
        [11, 13, 2, 5, 7, 12], [10, 3, 14, 11, 5, 13],
        [2, 1, 12, 9, 14, 12], [9, 4, 12, 14, 6, 5],
    ])
    singleton = greedy_select(LABELS, pool, max_size=1)
    assert singleton.selected_ids == ("candidate-2",)
    assert singleton.validation_auc == 5 / 6
    result = greedy_select(LABELS, pool, max_size=3)
    assert result.selected_ids == ("candidate-2", "candidate-0", "candidate-3")
    assert result.selected_indices == (2, 0, 3)
    assert [step.resulting_auc for step in result.steps] == [5 / 6, 8 / 9, 1]
    assert result.candidate_order == tuple(candidate.candidate_id for candidate in pool)
    assert result.validation_evaluations == 4 + 3 + 2
    assert pairwise_auc(LABELS, result.probabilities) == 1
    assert result == greedy_select(LABELS, pool, max_size=3)


def test_greedy_strict_stop_does_not_look_ahead():
    pool = candidates_from_grid([
        [12, 5, 9, 15, 9, 5], [15, 10, 7, 2, 5, 14], [14, 4, 4, 10, 12, 2],
    ])
    result = greedy_select(LABELS, pool, max_size=3)
    assert result.selected_ids == ("candidate-0",)
    assert result.validation_auc == 5 / 9
    assert result.stop_reason == "no_strict_improvement"
    assert not result.steps[-1].accepted
    assert pairwise_auc(LABELS, average_probabilities([c.probabilities for c in pool])) == 2 / 3


def test_greedy_no_replacement_and_same_member_different_epochs():
    pool = candidates_from_grid([[13, 6, 5, 5, 12, 14], [5, 12, 6, 10, 13, 4]])
    result = greedy_select(LABELS, pool, max_size=3)
    assert result.selected_ids == ("candidate-0", "candidate-1")
    assert result.validation_auc == 2 / 3
    assert result.stop_reason == "pool_exhausted"
    assert len({c.member_id for c in result.selected}) == 1


def test_greedy_ties_individual_auc_then_pool_order():
    pool = candidates_from_grid([[1, 1, 4, 4, 12, 8], [10, 15, 3, 14, 10, 13], [11, 3, 1, 10, 5, 9]])
    result = greedy_select(LABELS, pool, max_size=2)
    assert result.selected_ids == ("candidate-0", "candidate-2")
    assert result.validation_auc == 1
    # A lexicographically earlier ID must not override the supplied stable order.
    tied = [Candidate("z", "member", 0, (0.1, 0.9)), Candidate("a", "member", 1, (0.1, 0.9))]
    result = greedy_select([0, 1], tied, max_size=2)
    assert result.selected_ids == ("z",)
    assert not result.steps[-1].accepted
    assert greedy_select([0, 1], tied[::-1], max_size=2).selected_ids == ("a",)
    extension_tie = candidates_from_grid([
        [13, 6, 5, 5, 12, 14], [5, 12, 6, 10, 13, 4], [5, 12, 6, 10, 13, 4],
    ])
    extension_tie = (extension_tie[0], replace(extension_tie[1], candidate_id="z"),
                     replace(extension_tie[2], candidate_id="a"))
    assert greedy_select(LABELS, extension_tie, max_size=2).selected_ids == ("candidate-0", "z")


@pytest.mark.parametrize("probabilities,expected", [((0.5, 0.5), 0.5), ((0.9, 0.1), 0)])
def test_best_singleton_is_always_selected_even_without_skill(probabilities, expected):
    result = greedy_select([0, 1], [Candidate("a", "member", 0, probabilities)], max_size=4)
    assert result.selected_ids == ("a",)
    assert result.validation_auc == expected


def test_online_update_retains_incumbent_on_tie_or_regression():
    def select(name, probabilities):
        return greedy_select([0, 1], [Candidate(name, "m", 0, probabilities)], max_size=1)
    old = select("old", (0.4, 0.6))
    assert retain_best_selection(old, select("tied", (0.1, 0.9))) is old
    assert retain_best_selection(old, select("worse", (0.6, 0.4))) is old
    new = select("new", (0.1, 0.9))
    assert retain_best_selection(select("worse", (0.5, 0.5)), new) is new
    assert retain_best_selection(None, new) is new


def test_selection_validates_metadata_labels_and_limits():
    good = Candidate("a", "member", 0, [0.1, 0.9])
    for limit in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            greedy_select([0, 1], [good], max_size=limit)
    for labels, candidates in [([0, 1], []), ([0, 1], [good, good]), ([1, 1], [good]),
                               ([0, 1, 0], [good]), ([0, 2], [good])]:
        with pytest.raises(ValueError):
            greedy_select(labels, candidates, max_size=2)
    for kwargs in ({"candidate_id": ""}, {"member_id": ""}, {"source_epoch": True},
                   {"source_epoch": -1}, {"probabilities": [float("inf")]}):
        with pytest.raises(ValueError):
            replace(good, **kwargs)
    mutable = [0.1, 0.9]
    copied = Candidate("a", "member", 0, mutable)
    mutable[0] = 0.8
    assert copied.probabilities == (0.1, 0.9)
    with pytest.raises(TypeError):
        greedy_select([0, 1], [good], max_size=1, initial_selected=[good])


def test_numpy_boundaries_if_installed():
    np = pytest.importorskip("numpy")
    assert binary_roc_auc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5
    assert binary_log_loss(np.array([False, True]), np.array([0.1, 0.9])) == pytest.approx(-math.log(0.9))
    with pytest.raises(ValueError, match="one-dimensional"):
        binary_roc_auc(np.array([[0], [1]]), [0.1, 0.9])


def result_record(seed, auc=0.7, loss=0.4, method="mlp", **extra):
    return {
        "method": method, "seed": seed, "status": "complete",
        "metrics": {
            "test": {"roc_auc": auc, "log_loss": loss},
            "validation": {"roc_auc": auc - 0.1, "log_loss": loss + 0.1},
        },
        **extra,
    }


def test_mean_sample_std_null_zero_and_stability():
    assert mean_sample_std([]) == (None, None)
    assert mean_sample_std([0]) == (0, None)
    assert mean_sample_std([0, 0, 0]) == (0, 0)
    assert mean_sample_std([0.7, 0.8, 0.9]) == pytest.approx((0.8, 0.1))
    assert mean_sample_std([1e12, 1e12 + 2, 1e12 + 4]) == (1e12 + 2, 2)
    assert mean_sample_std([1e308, 1e308]) == (1e308, 0)
    for invalid in (True, "1", float("nan"), float("inf")):
        with pytest.raises(ValueError):
            mean_sample_std([invalid])


def test_aggregation_complete_golden_and_ordering():
    records = [result_record(seed, auc, loss, timing={"fit_wall_s": seconds})
               for seed, auc, loss, seconds in [(0, 0.6, 0.2, 10), (1, 0.7, 0.4, 14), (2, 0.8, 0.6, 18)]]
    expected = aggregate_results(records, expected_seeds=[0, 1, 2], expected_methods=["mlp"])
    group = expected["groups"]["mlp"]
    assert expected["complete"] and group["n_complete"] == 3
    assert group["metrics"]["test"]["roc_auc"]["mean"] == pytest.approx(0.7)
    assert group["metrics"]["test"]["roc_auc"]["sample_std"] == pytest.approx(0.1)
    assert group["metrics"]["test"]["log_loss"]["sample_std"] == pytest.approx(0.2)
    assert group["metrics"]["validation"]["roc_auc"]["mean"] == pytest.approx(0.6)
    assert group["timing"]["fit_wall_s"]["mean"] == 14
    assert group["timing"]["fit_wall_s"]["sample_std"] == 4
    for permutation in itertools.permutations(records):
        assert canonical_json(aggregate_results(permutation, expected_seeds=[2, 0, 1], expected_methods=["mlp"])) == canonical_json(expected)
    assert json.loads(canonical_json(expected)) == expected


def test_aggregation_optional_timing_does_not_drop_quality_and_members_are_not_runs():
    records = [result_record(0, timing={"fit_wall_s": 0}, members=[{"roc_auc": 1}] * 9), result_record(1)]
    group = aggregate_results(records)["groups"]["mlp"]
    assert group["metrics"]["test"]["roc_auc"]["n_observed"] == 2
    assert group["timing"]["fit_wall_s"] == {"n": 1, "n_observed": 1, "seeds": [0], "mean": 0, "sample_std": None}


def test_aggregation_missing_failed_cancelled_are_visible():
    records = [result_record(0), {"seed": 1, "method": "mlp", "status": "failed"},
               {"seed": 2, "method": "mlp", "status": "cancelled"}]
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_results(records, expected_seeds=[0, 1, 2, 3])
    summary = aggregate_results(records, expected_seeds=[0, 1, 2, 3], strict=False)
    assert not summary["complete"]
    group = summary["groups"]["mlp"]
    assert group["seeds"] == [0]
    assert group["missing_seeds"] == [3]
    assert group["incomplete_seeds"] == [1, 2, 3]
    assert summary["diagnostics"]["failed"] == [{"method": "mlp", "seed": 1}]
    assert summary["diagnostics"]["cancelled"] == [{"method": "mlp", "seed": 2}]
    assert group["metrics"]["test"]["roc_auc"]["sample_std"] is None


def test_aggregation_empty_requires_expected_matrix_and_is_incomplete():
    with pytest.raises(ValueError):
        aggregate_results([])
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_results([], expected_seeds=[0], expected_methods=["mlp"])
    summary = aggregate_results([], expected_seeds=[0], expected_methods=["mlp"], strict=False)
    metric = summary["groups"]["mlp"]["metrics"]["test"]["roc_auc"]
    assert not summary["complete"]
    assert metric["mean"] is None and metric["sample_std"] is None


@pytest.mark.parametrize("metric,bad", [("roc_auc", None), ("roc_auc", True), ("roc_auc", "0.8"),
    ("roc_auc", float("nan")), ("roc_auc", -0.1), ("roc_auc", 1.1),
    ("log_loss", None), ("log_loss", float("inf")), ("log_loss", -0.1)])
def test_aggregation_rejects_missing_or_invalid_complete_metrics(metric, bad):
    record = result_record(0)
    record["metrics"]["test"][metric] = bad
    with pytest.raises(ValueError):
        aggregate_results([record], strict=False)


def test_aggregation_rejects_duplicates_unexpected_runs_context_and_negative_time():
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_results([result_record(0), result_record(0)], strict=False)
    with pytest.raises(ValueError, match="unexpected seed"):
        aggregate_results([result_record(99)], expected_seeds=[0], strict=False)
    with pytest.raises(ValueError, match="unexpected method"):
        aggregate_results([result_record(0)], expected_methods=["other"], strict=False)
    with pytest.raises(ValueError, match="status"):
        aggregate_results([result_record(0, status="in_progress")], strict=False)
    with pytest.raises(ValueError, match="non-negative"):
        aggregate_results([result_record(0, timing={"fit_wall_s": -1})])
    with pytest.raises(ValueError, match="context"):
        aggregate_results([result_record(0, comparison_hash="a"), result_record(1, comparison_hash="b")])
    for version in (True, "1", 2):
        with pytest.raises(ValueError, match="schema_version"):
            aggregate_results([result_record(0, schema_version=version)])


def test_aggregation_methods_remain_separate_numeric_seeds_and_declared_order():
    records = [result_record(seed, auc, method=method) for method, auc in [("a", 0.9), ("b", 0.3)] for seed in (10, 2, 0)]
    result = aggregate_results(records, expected_methods=["b", "a"])
    assert list(result["groups"]) == ["b", "a"]
    assert result["groups"]["a"]["seeds"] == [0, 2, 10]
    assert result["groups"]["a"]["metrics"]["test"]["roc_auc"]["mean"] == 0.9
    assert result["groups"]["b"]["metrics"]["test"]["roc_auc"]["mean"] == 0.3


def test_paired_deltas_join_seed_ids_and_report_exclusions():
    paired = paired_deltas({0: 0.70, 1: 0.80, 2: 0.90}, {2: 0.90, 0: 0.72, 1: 0.81})
    assert paired["seeds"] == [0, 1, 2]
    assert paired["deltas"] == pytest.approx([0.02, 0.01, 0])
    assert paired["mean"] == pytest.approx(0.01)
    assert paired["sample_std"] == pytest.approx(0.01)
    partial = paired_deltas({0: 0.7, 1: 0.8}, {1: 0.9, 2: 1})
    assert partial["seeds"] == [1]
    assert partial["excluded_baseline_seeds"] == [0]
    assert partial["excluded_candidate_seeds"] == [2]
    assert partial["sample_std"] is None


def test_canonical_json_rejects_nonfinite_and_preserves_unicode_precision():
    value = {"z": 0.7123456789012345, "a": "é", "missing": None, "yes": True}
    assert json.loads(canonical_json(value)) == value
    assert canonical_json(value) == canonical_json(dict(reversed(list(value.items()))))
    with pytest.raises(ValueError):
        canonical_json({"bad": float("nan")})
