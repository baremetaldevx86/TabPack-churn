# Reduced Churn protocol

This document defines the local three-method experiment. Configuration,
preprocessing, model/optimizer, and selector details are grounded in
[`core.py`](../tabpack/core.py), [`data.py`](../tabpack/data.py),
[`models.py`](../tabpack/models.py), and [`optim.py`](../tabpack/optim.py).
Training/CLI orchestration is implemented in [`experiment.py`](../tabpack/experiment.py)
and [`cli.py`](../tabpack/cli.py). Consult [`checkpoints.md`](checkpoints.md) for
the verified implementation and execution status. Measured results are linked
from [`reports/churn-reduced/`](../reports/churn-reduced/); no illustrative
scores are mixed into the result artifacts.

## 1. Question and experimental unit

Compare three prespecified small MLP workflows on the same canonical Churn
train/validation/test split. A reporting observation is one **method × seed**
run. The default plan has three methods and seeds `[0, 1, 2]`: nine observations.
Members, batches, snapshots, and epochs are internal to a run, not additional
statistical replications.

This is a **reduced optimistic-protocol TabPack-style comparison**. The full
candidate sampling, training, and validation-selection procedure is repeated
for every packed-method seed. In the paper's **conservative** protocol, a main
run selects configurations and secondary runs retrain only the selected base
models; performance comes from those secondary runs while the main-run runtime
is charged. The local full reruns do not implement that conservative protocol.
"Optimistic" names this protocol distinction; it does not authorize test-driven
selection. Adaptive validation-score optimism is a separate issue.

## 2. Canonical data and fitting boundary

Source identity is pinned in [`data-source.json`](data-source.json):

- Dataset archive: `Yura52/tabpack-data`, revision
  `dfb85cb493bfb7493de0b8c0bd9e2f9e6a56baa1`, `tabpack-data.tar.gz`.
- Size: **189,856,645 bytes**.
- SHA-256:
  `89338c628fed24af03084c9348ca0a5c8ca4f12f8b988a576d9d1711cc661558`.
- Source split sizes: **6,400 train / 1,600 validation / 2,000 test**.
- Target: retain source int64 classes `0` and `1`; metrics use `1` as positive.
  Business semantics must come from source evidence, not an assumed remapping.

Acquisition downloads and verifies the complete archive, then publishes only
allowlisted Churn files to a new directory. Churn-only extraction does not mean
Churn-only network bytes. HTTPS redirects, archive paths/types, size/decompression
limits, schema, and split validity are checked. The loader uses
`np.load(..., allow_pickle=False)` after bounded NPY-header validation.

The archive is retained at `data/tabpack-data.tar.gz` for the default
`data/churn` destination. Every offline load verifies it and derives the exact
member sizes/digests from those bytes before comparing the extracted files.
An editable receipt or same-shaped local dataset cannot substitute for this
authority. Source indices stay in their original order and must be unique,
disjoint, in range, and collectively complete; the loader does not resplit.
Both classes must occur in each split. No alternative Churn data or synthetic
fallback is allowed in a real run.

### Train-only preprocessing

Fit one shared preprocessing state on train features, then transform every
split with that frozen state:

1. **Numeric:** train-median imputation, then scikit-learn
   `QuantileTransformer(output_distribution="normal")`. Use
   `n_quantiles=min(1000, n_train)`, `subsample=n_train`, and random state `0`.
   No noise is injected.
2. **Categorical:** train-most-frequent imputation, with ascending typed-value
   tie resolution, then dense float32 one-hot encoding with
   `handle_unknown="ignore"`. An unknown category yields zeros in that feature's
   block. Missing means `None` or floating NaN; `"nan"` and `""` remain categories.
3. **Explicit binary source blocks:** treat `x_bin` columns as categorical.
   Do not discover binary columns by scanning validation/test values.
4. Reject all-missing train columns; retain observed constant columns. Output
   numeric columns followed by categorical one-hot blocks as finite, contiguous
   float32. Row IDs stay outside the feature matrix.

The learned-state fingerprint covers medians, all quantile/reference arrays,
modes, category vocabularies, feature/block layout, parameters, and dependency
versions. Labels never enter fitting. `prepare_data` can validate/transform all
splits up front, but training and selection consume only train/validation
features and labels; final test scoring waits until selection is frozen.

This differs from the pinned official noisy-quantile and global binary-discovery
pipeline. Identical source splits do not make the preprocessing paper-equivalent.

## 3. Configuration and random streams

The serialized `ReducedConfig` and resolved `MemberSpec` records define each
run. Defaults are:

| Parameter | Value |
| --- | --- |
| Seeds | `[0, 1, 2]` |
| Ensemble base members / maximum selected snapshots | `4 / 4` |
| Maximum epochs / patience | `100 / 16` |
| Batch size | `256` per member, including correct reduction of the final short batch |
| Training dtype | `float32` |
| Fixed baseline depth / width / dropout | `2 / 64 / 0.1` |
| Fixed baseline AdamW learning rate / weight decay | `0.001 / 0.0001` |
| Heterogeneous depth choices | `[1, 2, 3]` |
| Heterogeneous width choices | `[32, 64, 96]` |
| Heterogeneous dropout choices | `[0.0, 0.1, 0.2]` |
| Heterogeneous learning-rate range | log-uniform `[0.0003, 0.003]` |
| Heterogeneous weight-decay range | log-uniform `[0.000001, 0.001]` |
| AdamW betas / epsilon | `(0.9, 0.999) / 1e-8` |
| Bias decay | `0` for every method |

Discrete choices are uniform. Named SHA-256-derived streams separate member
hyperparameter draws, initialization, permutations, and dropout. Sampling is
prefix-stable and independent of Python's randomized `hash()` and the global
random generator. Training must route actual draws through these resolved
streams, including member-specific epoch permutations; permuted labels must
follow the same indices as features. Record the seed/sampling algorithm versions
and the sampled member records instead of attempting to infer them from a root
seed later.

Each packed-method seed redraws its configurations. Identical seed integers do
not promise identical trajectories across architectures, pack sizes, devices,
library versions, or operating systems. There is no claim of CPU/CUDA bitwise
identity. The default recipe uses no feature embeddings, Muon, pruning,
learning-rate schedule, mixed precision, or external tuning.

## 4. Models and optimization

### Ordinary MLP

One fixed baseline network with hidden `Linear -> ReLU -> Dropout` blocks and
a linear binary-logit head. Depth counts hidden blocks; all hidden blocks within
one member use its configured width. Select the checkpoint with the best
validation ROC-AUC, retaining the earlier checkpoint on an exact tie.

### Homogeneous independent MLP ensemble

Four copies of the baseline architecture/settings, with separate parameters,
optimizer states, initialization, dropout, and data permutations. Update them
in a Python loop per batch. Evaluate the **mean of all four probabilities** each
epoch and checkpoint the four-member state jointly on strict validation-AUC
improvement. There is no per-member best-epoch mixing, greedy selection, or
individual pruning for this baseline. This baseline is neither the paper's
heterogeneous MLP-HPE nor a tuned SameHP/TabM reproduction.

### Heterogeneous packed ensemble

`PackedMLP` has member-first semantics: `[M, B, F]` inputs and `[M, B]` binary
logits; shared `[B, F]` inputs are also supported. Rectangular tensors allocate
the maximum member width and depth. Width padding is masked and missing-depth
blocks are skipped functionally. Members share an allocation, **not trainable
parameters**. Padding still consumes storage and padded-width arithmetic, so
logical parameter count and stored tensor count are distinct resources.

For member `m` with `B_m` examples, optimize:

```text
L_m = (1 / B_m) * sum_i BCEWithLogits(z_mi, y_mi)
L   = sum_m L_m
```

There is no division by member count and no loss on an averaged ensemble
prediction. PackedAdamW maintains independent moments and step counters per
parameter/member, with member-specific learning rates and weight decay.
Structural masks keep padding inert; the model-aware optimizer factory applies
zero bias decay. Ordinary models use stock PyTorch AdamW with the same bias
policy. A missing gradient and a present zero gradient have different AdamW
semantics. Optimizer pause/state-round-trip support is a tested primitive, not
a promise of resumable experiment training.

## 5. Online validation selection and stopping

For the packed method, after each epoch:

1. Form a stable candidate pool from **previously selected snapshots first**,
   in retained selection order, then current-epoch members in member order.
   Each candidate has a unique snapshot ID, member ID, source epoch, frozen
   weights, and validation probabilities.
2. Rebuild the greedy selection from scratch. Always select the best singleton,
   even when its AUC is at or below 0.5; there is no empty-ensemble baseline.
3. Score each unused candidate when added to the current probability average.
   Choose by prospective ensemble AUC, then the candidate's individual AUC,
   then stable pool order. Accept only an **exact strict** improvement.
4. The four-snapshot setting caps one greedy selection pass; it does not stop
   epoch training. Training stops only at the declared patience or 100-epoch
   limit.
5. Replace the incumbent ensemble only if the rebuilt validation AUC strictly
   improves. On a tie or regression, retain the entire previous selection.

Selection is without replacement by **snapshot ID**, not member ID. Distinct
epochs of one member can both be selected; four snapshots need not mean four
unique members. Copy their weights when snapshotting so later optimizer updates
cannot mutate the selected predictor. Averaging uses probabilities after
sigmoid, not logits or member AUCs.

The orchestration contract uses patience 16 for consecutive non-improving
validation epochs and a hard cap of 100. Record the exact stopping comparator,
epochs executed, selected source epochs, and stop reason in the run artifacts;
the paper's `patience + 1` convention must not be silently assumed. This draft
does not establish that the pending training implementation has verified that
boundary. Ordinary and homogeneous methods monitor their checkpoint scores;
the packed method monitors its retained ensemble score. There is no base-member
pruning. All decisions use validation, then the chosen predictor is frozen
before final test inference. Test scores cannot pick seeds, epochs, members,
thresholds, retries, or a revised protocol.

## 6. Metrics and reporting

- **Primary local metric:** ROC-AUC, higher is better. Equal-score cross-class
  pairs receive half credit; both classes are required. Checkpointing and
  greedy selection use validation ROC-AUC without rounded-score comparisons.
- **Secondary metric:** probability log loss, lower is better, using natural
  logarithms. The default metric clips at float64 epsilon `2**-52` solely for
  log loss; AUC uses the original scores.
- **Accuracy:** report separately at the fixed `0.5` probability threshold
  (`p >= 0.5` predicts class 1). The **official Churn score is accuracy**, not
  ROC-AUC. A local AUC must never be placed in an official-accuracy comparison
  column or described as matching a published Churn score. Local accuracy
  remains a changed-protocol result, even when the metric name matches.

Score ensemble probabilities, rather than averaging member-level metrics.
Report every planned method/seed result and arithmetic mean ± **sample** SD
(`ddof=1`). Members are not seeds. For one complete seed, SD is undefined/null;
for none, both mean and SD are null. Missing/failed/cancelled runs remain visible
and are never replaced with zero or dropped silently. A primary complete
comparison requires all nine planned observations.

Seed-paired differences are candidate minus baseline, joined on actual seed ID;
positive favors the candidate for AUC, negative for log loss or time. Report
paired seed membership. These are descriptive summaries: three seeds on one
split do not justify significance claims, p-values, or population-level ranking.
Do not choose a favorable retry or pool across incompatible dataset, protocol,
preprocessing, source, or timing identities.

## 7. Timing, capacity, and artifacts

Measure training/validation/selection separately from final frozen test
prediction. Use wall-clock timing with CUDA synchronization at the boundaries
when on GPU. Identify what each timer includes; data acquisition and archive
verification belong outside model-training time. Do not interpret a one-shot
final-inference timer as a repeated, warmed-up throughput benchmark. The report
must describe actual executed boundaries rather than borrow the research notes'
proposed benchmark repetition counts.

Record enough local provenance to audit a result:

- Actual local Git revision/dirty state, distinct from the pinned upstream
  reference; resolved configuration, method, seed, and member stream records.
- Canonical archive/member/split identity and preprocessing fingerprint.
- Python/package versions, platform, actual device, and relevant hardware.
- Per-seed metrics, executed/selected epochs, selected snapshot IDs and unique
  member count, stop reason, logical/stored parameter counts, and timing scope.
- Completion status and expected method/seed matrix; preserve failed attempts.

The source modules provide these primitives; complete artifact writing and
validation are integration responsibilities. Final checkpoint payloads are
weights and reconstruction metadata for **inference**, using
`torch.load(path, map_location="cpu", weights_only=True)`. Reconstruct each
recorded ordinary snapshot, load its state dict, enter evaluation mode, and
average probabilities as recorded. Preprocessing identity must match the
training run. There is **no exact resume**: final weights do not restore
optimizer, RNG, batch position, or selection history needed for continuation.

Raw data, model weights, and per-row prediction files remain local under
ignored `data/` and `runs/`. Published material in [`reports/`](../reports/)
contains aggregate results and suitable provenance, not row-level predictions.
The report command consumes saved run outcomes without training or downloading.

## 8. Command and verification boundary

From the checkout root with the selected environment's `python`:

```text
python -m tabpack.cli preflight --device cuda
python -m tabpack.cli fetch-data --data-dir data/churn
python -m tabpack.cli dry-run --data-dir data/churn --device cuda
python -m tabpack.cli smoke --output runs/smoke --device cpu
python -m tabpack.cli run --data-dir data/churn --device cuda --output runs/churn-reduced
python -m tabpack.cli report --run-dir runs/churn-reduced --output reports/churn-reduced
```

See [README](../README.md) for direct-venv PowerShell/Linux invocations.
`preflight` checks the requested device/environment; `dry-run` validates the
canonical local dataset and resolves preprocessing/configuration without an
optimizer training run. Only `fetch-data` acquires data. `smoke` explicitly
injects `synthetic_smoke_v1`; success is execution evidence, not Churn evidence.

CPU GitHub Actions uses Windows and Linux with Python 3.12, installs
`torch==2.11.0` from the CPU index followed by `requirements/base.txt`, then runs
`python -m pytest -m 'not gpu'` and `python -m ruff check tabpack tests`.
Acquisition tests mock transport and use generated archives; CI performs no
dataset downloads. Optional GPU or real-data checks must not turn ordinary test
collection into implicit acquisition. Actual test receipts belong in
[`checkpoints.md`](checkpoints.md), not an inferred pass badge.

## 9. Interpretation limits and references

The methods share splits, preprocessing, epoch caps, and per-member batch size.
They are **not capacity-, compute-, wall-time-, or validation-search-matched**.
One trajectory versus four, heterogeneous widths/depths, padded execution,
joint checkpointing, and greedy snapshot selection all change the comparison.
An accuracy/AUC difference cannot be attributed solely to heterogeneity; a
runtime difference cannot be attributed solely to packing. Laptop/CPU timings
do not reproduce the paper's hardware efficiency.

The small candidate pool, three seeds, fixed untuned baseline, altered batch
size/preprocessing, AdamW-only optimization, and absence of embeddings/pruning
are material reductions. Reusing validation for checkpointing, early stopping,
and selection can overfit that split. One fixed Churn split does not establish
performance on other splits or datasets. If test results guide development,
subsequent results on that same test set remain exploratory; another seed does
not create an untouched holdout. A failure to beat a baseline is a valid result.

Reference material was read, not imported, executed, or vendored:

- [TabPack paper v1](https://arxiv.org/html/2607.05380v1): packed training and
  selection (§3), evaluation protocols (§4.2), validation overfitting (Appendix C),
  and data/experiment details (Appendices E–G).
- [Official source at `05a89e21b955f12de84889d662e15ca534019aaa`](https://github.com/yandex-research/tabpack/tree/05a89e21b955f12de84889d662e15ca534019aaa).
- Local implementation handoffs: [T51](agents/T51.md), [T52](agents/T52.md),
  [T53](agents/T53.md); protocol/review evidence: [T4](agents/T4.md),
  [T29](agents/T29.md), [T32](agents/T32.md).
