# Checkpoints and integration history

## C0: preflight and research

- Empty local Git repository on `master`; existing user commit identity available.
- Requested destination: `git@github.com:baremetaldevx86/TabPack-churn.git`.
- Correct official reference: `https://github.com/yandex-research/tabpack`, pinned
  to `05a89e21b955f12de84889d662e15ca534019aaa`. Supplied unhyphenated URL returned 404.
- Python 3.12.10, Windows AMD64, i5-13420H, 16 GiB RAM, RTX 5050 Laptop 8 GiB.
- Global CUDA wheel probe succeeded, but experiment installation is isolated.
- Fifty distinct initial assignments attempted in parallel. T49 had an invalid
  dispatch argument and was dispatched correctly afterwards; several callbacks
  were interrupted. Existing notes are preserved; recovery assignments close
  unfinished evidence instead of claiming all callbacks succeeded.
- All agents are prohibited from Git writes: coordinator reviews and serializes
  focused integration commits. Agent notes are proposals, not implementation or
  test-pass evidence. Source/testing follow in separate checkpoints.
- Scientific choices: three seeds, four members, AdamW, no embeddings/pruning,
  validation ROC-AUC selection, unchanged canonical splits. Accuracy is reported
  separately because the official Churn task metric is accuracy.

## C1: implementation, environment, and data validation

- Added independently authored `tabpack/core.py`, `data.py`, `models.py`,
  `optim.py`, `experiment.py`, and `cli.py`, plus focused tests and the
  PowerShell/Linux documentation. No upstream modules are imported or executed.
- Isolated `.venv` uses Python 3.12.10, NumPy 2.2.6, SciPy 1.15.3,
  scikit-learn 1.6.1, PyTorch 2.11.0+cu128, pytest 8.4.2, and Ruff 0.12.12.
  `pip check` passed. CUDA preflight passed on an RTX 5050 Laptop (compute 12.0,
  bundled CUDA 12.8, driver 610.88); CPU is supported separately.
- Focused/full checks: `python -m pytest -q` -> **296 passed, 1 skipped**;
  `python -m ruff check tabpack tests` -> **passed**. The single skip is an
  unprivileged Windows symlink test. The sparse-gradient test emits one upstream
  PyTorch warning while passing.
- Synthetic smoke: `python -m tabpack.cli smoke --output runs/smoke --device cpu`
  and report generation completed for all three methods. Smoke artifacts are
  synthetic engineering evidence only and are not mixed with Churn results.
- The pinned 189,856,645-byte archive was downloaded and SHA-256 verified as
  `89338c628fed24af03084c9348ca0a5c8ca4f12f8b988a576d9d1711cc661558`.
  Extraction initially exposed two real archive-format facts: AppleDouble
  `._*` sidecars and a non-Churn Microsoft member larger than the scanner's
  Churn per-member limit. The bounded scanner now validates/skips sidecars and
  streams non-Churn members without loading them. Canonical dry-run passed with
  6,400/1,600/2,000 rows, raw 7/3/1 feature blocks, and 16 prepared features.

## C2: measured reduced Churn comparison

- Command: `python -m tabpack.cli run --data-dir data/churn --device cuda --output runs/churn-reduced`.
- Report: `reports/churn-reduced/report.md`, `per_seed.csv`, and `summary.json`.
- All nine method×seed runs completed; no failures/cancellations/missing seeds.
  The selected device was CUDA on the RTX 5050 Laptop. Test ROC-AUC mean ±
  sample SD: ordinary MLP **0.855624 ± 0.001718**, homogeneous independent
  ensemble **0.856576 ± 0.000944**, packed heterogeneous ensemble
  **0.858702 ± 0.002105**. Test log-loss: **0.346776 ± 0.002816**,
  **0.344509 ± 0.001083**, and **0.343824 ± 0.004010**, respectively.
- Mean fit wall time in the same environment was 15.1349 s, 65.6732 s, and
  84.2583 s, respectively. These are complete workflow timings, not an isolated
  packing speedup. The packed method's mean was affected by seed-to-seed runtime
  variation; inspect `per_seed.csv` before quoting ratios.
- The local run was executed before the first local Git commit, so its manifest's
  `source.local_commit` is `null` with reason `unborn_repository`. The final
  repository commit records the implementation and report artifacts; rerunning
  with a committed checkout is the stricter provenance path. The upstream
  reference remains pinned separately to `05a89e21b955f12de84889d662e15ca534019aaa`.

## C3: release-candidate integrity fixes

- T57's intentional report-integrity failure was fixed: reports now reject
  result/manifest comparison-hash mismatches, missing/out-of-tree checkpoints,
  and checkpoint SHA-256 mismatches. The focused runner/CLI suite is now
  **7 passed**; the full suite is **303 passed, 1 skipped** and Ruff passes.
- The runner rejects unsupported `float64` configuration instead of silently
  executing float32, hashes the actual method subset/source identity into the
  comparison identity, records actual ordinary/homogeneous best checkpoint
  epochs, includes snapshot lineage fields in inference checkpoints, and times
  construction plus fit/selection under the synchronized fit boundary.
- The pre-commit Churn numbers remain retained as historical evidence only until
  the reviewed implementation is committed and the exact three-seed run is
  rerun against that commit. Do not call the pre-commit table the final release
  result before C4.

## C4: final committed-checkout Churn evidence

- Commit used by the run: `0f61be3ba7404d54bcde79ab5c92d87601426fd2`.
- Command: `python -m tabpack.cli run --data-dir data/churn --device cuda --output runs/churn-reduced-committed`.
- Report: `reports/churn-reduced-committed/report.md`, `per_seed.csv`, and
  `summary.json`; all nine runs completed and report-side checkpoint/hash
  validation passed.
- Final committed-checkout test ROC-AUC mean ± sample SD: ordinary MLP
  **0.855414 ± 0.002412**, homogeneous independent ensemble
  **0.856576 ± 0.000944**, packed heterogeneous ensemble
  **0.858825 ± 0.002562**. Test log-loss: **0.345809 ± 0.003415**,
  **0.344509 ± 0.001083**, and **0.343488 ± 0.004115**, respectively.
- Mean synchronized fit wall time was 9.6310 s, 50.2010 s, and 60.9600 s;
  these include construction plus training/validation/selection under the
  corrected boundary and remain complete workflow timings, not a packing-only
  speed claim. The packed result's seed-2 selection used two snapshots because
  additional candidates did not strictly improve validation AUC.

### Review decisions

- T3/T23's paper-faithful greedy singleton rule supersedes T16's proposed empty
  ensemble baseline: always start from the best singleton; rebuild from scratch
  each epoch from current members plus prior selected snapshots. Keep the prior
  ensemble on an epoch-level tie. No empty final predictor.
- Homogeneous ensemble: four independent MLPs updated in a Python loop per batch,
  all-member probability average, checkpoint and early stop on ensemble validation
  AUC. No individual-member pruning or greedy selection for this baseline.
- Depth padding is skipped functionally; padded weights are inert. Biases have
  zero AdamW decay in every method. Per-member sample-mean losses are summed.
- Source implementation is intentionally compact. Final inference checkpoints
  are required; exact mid-run resume is out of scope and must not be advertised.
- Training and final inference are timed separately with CUDA synchronization;
  timings are local workflow costs, not a packing-only speed benchmark.
