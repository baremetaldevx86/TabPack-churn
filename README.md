# Reduced TabPack-style Churn experiment

An independently authored, small comparison of three MLP workflows on the
canonical Churn split:

1. **Ordinary MLP:** one fixed-configuration network.
2. **Homogeneous independent ensemble:** four separately parameterized MLPs,
   trained in a Python loop, with an all-member probability average.
3. **Heterogeneous packed ensemble:** four independently configured networks in
   member-first tensors, with validation-only online greedy snapshot selection.

This is a **reduced, optimistic-protocol TabPack-style experiment**. It uses
three seeds, AdamW, and small networks; it is not a reproduction of the paper's
scores or an equal-capacity/equal-compute comparison. The official Churn metric
is **accuracy**; local checkpoint and ensemble selection use **ROC-AUC**.

**Measured result (RTX 5050 Laptop, CUDA 12.8 wheel, three seeds):** the packed
heterogeneous workflow reached test ROC-AUC **0.858825 ± 0.002562**, compared
with **0.856576 ± 0.000944** for the homogeneous independent ensemble and
**0.855414 ± 0.002412** for the ordinary MLP. Test log-loss was
**0.343488 ± 0.004115**, **0.344509 ± 0.001083**, and **0.345809 ± 0.003415**,
respectively. These are descriptive reduced-protocol results, not a claim of
paper equivalence. The complete per-seed table is in
[`reports/churn-reduced-committed/report.md`](reports/churn-reduced-committed/report.md) and
[`reports/churn-reduced-committed/per_seed.csv`](reports/churn-reduced-committed/per_seed.csv).
See [integration history](docs/checkpoints.md) for checks actually completed.

> **Implementation status:** `core`, `data`, `models`, `optim`, `experiment`, and
> `cli` are implemented. The command sequence below was exercised for the
> synthetic smoke and the canonical Churn comparison. See the integration
> history for the exact test and acquisition receipts.

## Requirements

- 64-bit **Python 3.12** on Windows or Linux.
- An isolated virtual environment with [`requirements/base.txt`](requirements/base.txt):
  NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.6.1, setuptools 80.9.0,
  pytest 8.4.2, and Ruff 0.12.12.
- **PyTorch 2.11.0**, installed separately from the explicit CPU or CUDA 12.8
  wheel index. CUDA execution also needs a compatible NVIDIA GPU and driver.

The CUDA wheel supplies the runtime libraries needed here. A full CUDA Toolkit,
`nvcc`, custom CUDA extension build, upstream environment, and upstream code
execution are unnecessary. Wheel installation alone does not prove GPU readiness;
use the CUDA preflight below. These are version pins, not a fully hashed lock of
every transitive dependency; retain the resolved environment with run provenance.

## Windows PowerShell quickstart

Run from the repository root in PowerShell 5.1 or newer. Invoke the venv's
executable directly, so activation and execution-policy changes are unnecessary.
Run each stage only after the previous stage succeeds.

### 1. Create a fresh environment

```powershell
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath '.\pyproject.toml')) { throw 'Run from the repository root.' }
if (Test-Path -LiteralPath '.\.venv') { throw 'A .venv already exists; use or inspect it before recreating it.' }
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
$Python = (Resolve-Path -LiteralPath '.\.venv\Scripts\python.exe').Path
& $Python --version
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment Python is unavailable.' }
```

For a new PowerShell session using an already prepared environment, set
`$Python = (Resolve-Path -LiteralPath '.\.venv\Scripts\python.exe').Path` again.

### 2. Install one PyTorch build, then the common requirements

Choose **one** of these PyTorch commands for the fresh environment.

CUDA 12.8:

```powershell
& $Python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
```

CPU:

```powershell
& $Python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
```

Then install the shared dependencies from the normal package index:

```powershell
& $Python -m pip install -r requirements/base.txt
if ($LASTEXITCODE -ne 0) { throw 'Common dependency installation failed.' }
& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
```

All commands use the source checkout from its root; an editable installation is
not required for `python -m tabpack.cli`.

### 3. Check readiness and acquire the canonical data

The following sequence uses CUDA for the real experiment. For a CPU experiment,
replace `--device cuda` with `--device cpu` in preflight, dry-run, and run.
An explicit CUDA request must succeed on CUDA; it must not silently fall back.

```powershell
& $Python -m tabpack.cli preflight --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed; resolve its diagnostics before continuing.' }
& $Python -m tabpack.cli fetch-data --data-dir data/churn
if ($LASTEXITCODE -ne 0) { throw 'Canonical data acquisition failed.' }
& $Python -m tabpack.cli dry-run --data-dir data/churn --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Data/configuration readiness check failed.' }
```

`fetch-data` downloads the **full pinned archive** (189,856,645 bytes, about
181 MiB), verifies its size and SHA-256, and extracts only Churn. Keep
`data/tabpack-data.tar.gz`: offline loading uses it to verify the exact local
member bytes. `data/churn` means the final dataset directory. It should not be
pre-created for extraction; the data API publishes a fresh destination and
refuses to overwrite one. If the canonical dataset already exists, skip
acquisition and use `dry-run` to validate it.

The source and validation policy are in [`docs/data-source.json`](docs/data-source.json).
The loader preserves the original 6,400/1,600/2,000 train/validation/test rows
and index order. It never substitutes another Churn dataset or synthetic data.
Acquisition is explicit; preflight, dry-run, smoke, run, and report do not
download experiment data.

### 4. Check the execution path with a tiny synthetic smoke run

```powershell
& $Python -m tabpack.cli smoke --output runs/smoke --device cpu
if ($LASTEXITCODE -ne 0) { throw 'Synthetic smoke failed.' }
```

This uses the separate `synthetic_smoke_v1` fixture (32/12/12 rows), small
budgets, and all three methods. It needs no Churn download and can also be run
before acquisition. Keep its artifacts separate from Churn results.

### 5. Run and report the reduced experiment

```powershell
& $Python -m tabpack.cli run --data-dir data/churn --device cuda --output runs/churn-reduced
if ($LASTEXITCODE -ne 0) { throw 'Experiment did not complete; inspect the run artifacts.' }
& $Python -m tabpack.cli report --run-dir runs/churn-reduced --output reports/churn-reduced
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }
```

The same report step is available as a standalone script. It writes one JSON
summary, one per-seed CSV, and one Markdown report without training or data
acquisition:

```powershell
& $Python scripts/report_results.py `
  --run-dir runs/churn-reduced-committed `
  --output reports/churn-reduced-committed
```

Use a fresh output path for a new attempt. There is **no exact training-resume
interface**; a final inference checkpoint is not an interrupted-training state.
`report` reads saved results rather than training again. Inspect every seed and
completion status before interpreting a summary.

## Linux setup and equivalent commands

From the checkout root, use a fresh `.venv`. This example uses the CPU wheel;
select the `cu128` index instead for a separately prepared CUDA environment.

```bash
set -e
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements/base.txt
.venv/bin/python -m pip check
.venv/bin/python -m tabpack.cli preflight --device cpu
.venv/bin/python -m tabpack.cli smoke --output runs/smoke --device cpu
.venv/bin/python -m tabpack.cli fetch-data --data-dir data/churn
.venv/bin/python -m tabpack.cli dry-run --data-dir data/churn --device cpu
.venv/bin/python -m tabpack.cli run --data-dir data/churn --device cpu --output runs/churn-reduced
.venv/bin/python -m tabpack.cli report --run-dir runs/churn-reduced --output reports/churn-reduced
```

## Default experiment

[`ReducedConfig`](tabpack/core.py) is the source of the resolved defaults.

| Setting | Reduced default |
| --- | --- |
| Run seeds | `0, 1, 2` |
| Base members | 1 ordinary MLP; 4 homogeneous or packed members |
| Maximum selected packed snapshots | 4 |
| Maximum epochs / patience | 100 / 16 |
| Batch size | 256 **per member** |
| Training dtype | float32 |
| Ordinary/homogeneous architecture | 2 hidden layers, width 64, dropout 0.1 |
| Ordinary/homogeneous AdamW | learning rate 0.001, weight decay 0.0001 |
| Packed depth / width / dropout choices | `{1,2,3}` / `{32,64,96}` / `{0,0.1,0.2}` |
| Packed learning rate | log-uniform `[0.0003, 0.003]` |
| Packed weight decay | log-uniform `[0.000001, 0.001]` |
| Bias weight decay | 0 for every method |

Numeric median imputation and quantile-normal transformation are fit **only on
train**. Categorical mode imputation and one-hot encoding are also train-fitted;
unknown held-out categories map to zero in their one-hot block. All methods
share the same prepared data. There are no feature embeddings, Muon, base-model
pruning, or external hyperparameter tuning. See the [full protocol](docs/protocol.md)
for loss reduction, independent RNG streams, selection, metrics, and limitations.

## Source map

| Module | Responsibility |
| --- | --- |
| [`tabpack/core.py`](tabpack/core.py) | Immutable configuration/member records, named seed derivation and sampling, tie-aware ROC-AUC, log loss, probability averaging, greedy selection, and descriptive aggregation. Standard-library only. |
| [`tabpack/data.py`](tabpack/data.py) | Pinned acquisition, bounded Churn-only extraction, offline byte/schema/split validation, train-only preprocessing and its fingerprint, explicit synthetic fixture. |
| [`tabpack/models.py`](tabpack/models.py) | Ordinary MLP and heterogeneous member-first packed MLP, width/depth masks, member snapshots, logical/stored parameter counts, sample-mean/member-sum BCE loss. |
| [`tabpack/baselines.py`](tabpack/baselines.py) | Public reusable `SimpleMLP` baseline with configurable hidden width/depth/dropout and logits-first output. |
| [`tabpack/optim.py`](tabpack/optim.py) | Member-separable packed AdamW; optimizer factory applies structural masks and zero bias decay. Ordinary models use stock PyTorch AdamW. |
| [`tabpack/experiment.py`](tabpack/experiment.py) | Three-method training, early stopping, frozen snapshot/test evaluation, timing, local inference artifacts, and report orchestration. |
| [`tabpack/cli.py`](tabpack/cli.py) | `preflight`, `fetch-data`, `dry-run`, `smoke`, `run`, and `report` command boundary. |

Final checkpoints contain inference weights and reconstruction metadata, loaded
with `torch.load(..., map_location="cpu", weights_only=True)`. They do not promise
optimizer/RNG restoration or exact continuation. Per-row predictions, model
weights, and raw data stay **local** under ignored `runs/` and `data/`; publish
aggregate reports and appropriate provenance in `reports/`, not prediction files.

## Tests and CPU CI

Windows, using the same `$Python` variable:

```powershell
& $Python -m pytest -m 'not gpu'
if ($LASTEXITCODE -ne 0) { throw 'CPU test suite failed.' }
& $Python -m ruff check tabpack tests
if ($LASTEXITCODE -ne 0) { throw 'Ruff checks failed.' }
```

Linux uses `.venv/bin/python` for the same two module commands. The
[GitHub Actions workflow](.github/workflows/tests.yml) runs Python 3.12 on
Windows and Linux, explicitly installs the CPU PyTorch wheel before the common
requirements, and runs these checks. Test execution uses synthetic fixtures and
mocked acquisition; it performs no dataset downloads or measured experiments.
Dependency installation itself needs network access. CUDA correctness tests are
optional and excluded by `-m 'not gpu'`.

The tests cover independent packed-forward/gradient and AdamW oracles, padding
and member isolation, train-only preprocessing, source integrity, metrics,
deterministic sampling/selection, and aggregation. A passing CPU suite does not
establish CUDA performance, canonical acquisition success, or paper equivalence.

## Troubleshooting

- **CUDA unavailable:** inspect preflight and confirm the selected venv, CUDA
  wheel, GPU, and driver. Choose CPU explicitly if that is the intended run and
  retain the actual device in its provenance.
- **Missing preprocessing dependency:** install `requirements/base.txt` with
  the same venv interpreter used for execution.
- **Data missing or integrity mismatch:** retain the pinned archive next to
  `churn`, keep the source directory unmodified, and validate with `dry-run`.
  Extraction refuses an existing destination; use a fresh data directory when
  reacquiring rather than merging files or bypassing a hash failure.
- **`tabpack.cli` missing:** the checkout has not yet incorporated CLI
  integration, or the command was run outside the checkout root. Consult
  [checkpoints](docs/checkpoints.md) and the source map.
- **Run interrupted or report incomplete:** retain the failed attempt for
  diagnosis, choose a new run output for a restart, and do not represent a
  subset of seeds as the full experiment.

## Interpretation and references

Report all prespecified seeds and **mean ± sample SD (`ddof=1`)**. Three seeds
on one fixed split support descriptive local observations, not statistical
significance or broad ranking claims. Repeated validation selection can overfit
validation; test predictions are produced only after the predictor is frozen.
Equal epoch caps do not match capacity, compute, or validation-search effort.
Local timings measure complete workflows, not the isolated speed benefit of
packing. The [protocol](docs/protocol.md) explains the paper's optimistic versus
conservative evaluation and the reduced method's deviations.

- [TabPack paper, v1](https://arxiv.org/html/2607.05380v1).
- [Official reference repository, pinned revision `05a89e2`](https://github.com/yandex-research/tabpack/tree/05a89e21b955f12de84889d662e15ca534019aaa).
- [Pinned data source and preprocessing policy](docs/data-source.json).
- [Integration/checkpoint receipts](docs/checkpoints.md) and [task ledger](swarm.md).

The paper and pinned official source are reference material. This project does
not import, execute, or vendor the upstream implementation.
