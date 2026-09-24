# Goal

Build an independent, small, reproducible Churn-only experiment comparing an ordinary MLP, a homogeneous independent MLP ensemble, and a reduced heterogeneous packed TabPack-style ensemble. Consult the paper and pinned official code for reference only. Deliver tested implementation, real local measurements, documentation, and focused commits in the user's repository. Fifty independently delegated assignments communicate through this ledger and their task notes; the coordinator integrates and verifies all work.

## Tasks

| id | task | owner | status | result |
| --- | --- | --- | --- | --- |
| T1 | Paper Churn protocol and dataset identity | agent-01 | done | Recovered in docs/agents/T1.md by T54 after interrupted callback; identity, split, target and accuracy-vs-AUC boundary documented. |
| T2 | Paper packed architecture mathematics review | agent-02 | done | Documentation/research note complete in docs/agents/T2.md; no implementation tests run. |
| T3 | Paper online greedy selection semantics | agent-03 | done | Wrote docs/agents/T3.md: pinned paper/code semantics for online greedy pools, strict improvement, deterministic ties, snapshots, and ROC-AUC-vs-official-Churn accuracy. |
| T4 | Paper evaluation protocols and fair reductions | agent-04 | done | Wrote docs/agents/T4.md: paper optimistic/conservative protocols, fair reduced three-way Churn contract, resource/attribution caveats, and reporting rules. |
| T5 | Official Churn config and hyperparameters review | agent-05 | done | Recovered in docs/agents/T5.md by T54 after interrupted callback; pinned config and reduced deviations documented. |
| T6 | Official Churn reported results extraction | agent-06 | done | Documentation/research note complete in docs/agents/T6.md; no local training/tests run. |
| T7 | Churn-only download source and provenance review | agent-07 | done | Recovered in docs/agents/T7.md by T54 after interrupted callback; source identity and full-archive trust boundary documented; archive still unverified. |
| T8 | Official preprocessing audit | agent-08 | done | Audited pinned official Churn preprocessing and documented train-only fit plus binary-discovery/cache caveats in docs/agents/T8.md. |
| T9 | Official optimizer behavior audit | agent-09 | done | Wrote docs/agents/T9.md: pinned paper/source optimizer audit, AdamW-only Windows reduction, Muon risks, and equivalence-test guidance. |
| T10 | Official licensing and attribution review | agent-10 | done | Recovered in docs/agents/T10.md by T54 after interrupted callback; attribution and unresolved license evidence documented. |
| T11 | Windows CUDA compatibility review | agent-11 | done | Documentation/probe review complete in docs/agents/T11.md; no project tests/training run. |
| T12 | Dependency and reproducibility strategy review | agent-12 | done | Documentation/design note complete in docs/agents/T12.md; no install or lock validation run. |
| T13 | Configuration and member sampling design review | agent-13 | done | Documentation/design note complete in docs/agents/T13.md; no source or automated tests created by T13. |
| T14 | Ordinary and packed model design review | agent-14 | done | Documentation/design note complete in docs/agents/T14.md; implementation was later owned by T51. |
| T15 | Packed AdamW design review | agent-15 | done | Documentation/design note complete in docs/agents/T15.md; implementation was later owned by T51. |
| T16 | Greedy ensemble selection design review | agent-16 | done | Documentation/design note complete in docs/agents/T16.md; implementation was later owned by T53. |
| T17 | Binary metric implementation design review | agent-17 | done | Documentation/design note and standalone oracle checks complete in docs/agents/T17.md; no production metric source was created by T17. |
| T18 | Dataset validation and preprocessing design review | agent-18 | done | Documentation/design note complete in docs/agents/T18.md; implementation was later owned by T52. |
| T19 | Safe preflight diagnostic design review | agent-19 | done | Documentation/design note and bounded host probes complete in docs/agents/T19.md; no production preflight source was created by T19. |
| T20 | Results aggregation design review | agent-20 | done | Documentation/design note complete in docs/agents/T20.md; implementation remained coordinator-owned. |
| T21 | Packed model equivalence test design | agent-21 | done | Documentation/test design note complete in docs/agents/T21.md; executable tests were later owned by T51. |
| T22 | Packed optimizer equivalence test design | agent-22 | done | Documentation/test design note complete in docs/agents/T22.md; executable tests were later owned by T51. |
| T23 | Greedy selection test design | agent-23 | done | Documentation/test design note complete in docs/agents/T23.md; executable core tests were later owned by T53. |
| T24 | Data leakage and preprocessing test design | agent-24 | done | Wrote docs/agents/T24.md with adversarial train-only preprocessing, unknown-category, lineage, cache, and split-contamination test design. |
| T25 | Metric correctness test design/reconciliation | agent-25 | done | Recovered in docs/agents/T25.md by T54 after interrupted callback; current T53 metric tests and receipts reconciled, no separate T25 source file. |
| T26 | Config and sampling test design | agent-26 | done | Wrote docs/agents/T26.md with config invalid-range/type cases, deterministic sampling invariants, log-uniform oracle, and T13/T34 coordination questions. |
| T27 | Architecture documentation | agent-27 | done | Drafted docs/agents/T27.md with architecture diagrams, packed tensor/loss semantics, official-status boundaries, test status, and coordination questions. |
| T28 | Reproduction guide draft | agent-28 | done | Drafted preflight, isolated install, acquisition/validation, synthetic smoke plus Churn readiness dry run, reduced run, inspection, and troubleshooting guidance in docs/agents/T28.md. |
| T29 | Scientific limitations documentation | agent-29 | done | Wrote docs/agents/T29.md with reduced Churn validity threats and claim boundaries. |
| T30 | Dataset card documentation | agent-30 | done | Recovered in docs/agents/T30.md by T54 after interrupted callback; card documents observed, unobserved and unresolved license facts. |
| T31 | CI workflow design review | agent-31 | done | Documentation/design note complete in docs/agents/T31.md; no workflow/source changes. |
| T32 | Statistical reporting review | agent-32 | done | Documentation/review note complete in docs/agents/T32.md; no tests/training run. |
| T33 | Loss independence review | agent-33 | done | Wrote docs/agents/T33.md: member-separable training, sample-mean/member-sum BCE, AdamW scaling caveats, and gradient/isolation oracle guidance. |
| T34 | RNG independence review | agent-34 | done | Recovered in docs/agents/T34.md by T54 after interrupted callback; core stream policy and model seam reconciled, trainer proof remains pending. |
| T35 | Checkpoint design review | agent-35 | done | Documentation/design note complete in docs/agents/T35.md; no checkpoint implementation/tests run. |
| T36 | Numerical stability review | agent-36 | done | Documentation/probe review complete in docs/agents/T36.md; no project tests/training run. |
| T37 | Benchmark timing design review | agent-37 | done | Authored docs/agents/T37.md with reproducible wall-clock/CUDA-event boundaries, warmups, synchronization, and fair reporting rules. |
| T38 | Download security design review | agent-38 | done | Wrote docs/agents/T38.md with HTTPS/redirect, pinned digest, Windows-safe extraction, archive limits, partial-verification, and no-substitution requirements. |
| T39 | Local Git release workflow review | agent-39 | done | Added docs/agents/T39.md: unborn-repository checkpoints, exact-run SHA preservation, remote/push verification, and coordinator receipt requirements. |
| T40 | Hyperparameter fairness review | agent-40 | done | Recovered in docs/agents/T40.md by T54 after interrupted callback; budgets, confounds and fair-attribution boundaries reconciled. |
| T41 | CLI acceptance specification | agent-41 | done | Wrote docs/agents/T41.md with preflight/dry-run/run/report contract, options, exit codes, and acceptance gates. |
| T42 | End-to-end test design | agent-42 | done | Added deterministic offline synthetic all-path smoke specification in docs/agents/T42.md. |
| T43 | Preflight test design | agent-43 | done | Documentation/test design note complete in docs/agents/T43.md; no executable preflight tests created by T43. |
| T44 | Results aggregation test design | agent-44 | done | Documentation/test design note complete in docs/agents/T44.md; executable aggregation tests were later owned by T53. |
| T45 | Independent training oracle test design | agent-45 | done | Documentation/test design note complete in docs/agents/T45.md; no executable oracle created by T45. |
| T46 | Model resource accounting review | agent-46 | done | Documentation/review note complete in docs/agents/T46.md; no resource benchmark run. |
| T47 | Validation selection overfitting review | agent-47 | done | Documented validation-selection optimism, frozen untouched-test evaluation, all-seed reporting, and honest holdout handling in docs/agents/T47.md. |
| T48 | Provenance artifact specification | agent-48 | done | Defined a secret-free provenance manifest covering Git/source, URLs, dependencies, hardware/config/data hashes, timestamps, timing, checkpoints, and acceptance checks. |
| T49 | Portable PowerShell quickstart review | agent-49 | done | Wrote docs/agents/T49.md with a path-safe PowerShell 5.1 quickstart, locked venv install, preflight/acquisition/dry-run/run gates, and T12/T19/T28/T38/T41/T48 reconciliation. |
| T50 | Scientific acceptance and release gate | agent-50 | done | Added docs/agents/T50.md with mandatory tests, reproducibility, provenance, completeness, docs, Git, and limitations gates. |
| T51 | Implement ordinary/packed models and AdamW with oracle tests | model-integrator | done | Implemented owned models/AdamW/tests and docs/agents/T51.md; last completed focused run 78 passed including 2 CUDA tests; later torch-import stall recorded. |
| T52 | Implement canonical download, loader, preprocessing and tests | data-integrator | done | Implemented tabpack/data.py, tests/test_data.py, docs/data-source.json and docs/agents/T52.md; focused suite 70 passed, 9 skipped (8 missing sklearn, 1 Windows symlink privilege). |
| T53 | Implement configuration, metrics and greedy selection with tests | core-integrator | done | Implemented tabpack/core.py, tests/test_core.py, docs/agents/T53.md; focused suite 140 passed, including 8,604 AUC oracle vectors. |
| T54 | Recover interrupted research and reconcile 50-agent ledger | recovery-reviewer | done | Reconciled 50/50 original notes (42 found, 8 recovered), corrected design/review titles, and recorded source/metric/license/trust-boundary findings in docs/agents/T54.md. |
| T55 | Documentation, CI and experiment review | release-writer | done | Added README.md, docs/protocol.md, and .github/workflows/tests.yml; documented integration-pending CLI caveats and found core.py Ruff I001. |
| T56 | Independent runner correctness and release review | runner-reviewer | done | Wrote docs/agents/T56.md: dtype, timing, interruption, artifact-integrity, snapshot-lineage, provenance, and release-claim blockers; confirmed T57's hash-tamper gap. |
| T57 | Runner and CLI integration regression tests | integration-tester | done | Added focused synthetic runner/CLI integration tests; 6 passed, 1 exposed missing report comparison-hash tamper rejection. See docs/agents/T57.md. |

## Findings

- Coordinator: empty local repository, no initial commits/remotes; remote SSH query succeeded with no refs. Python 3.12.10, Windows, i5-13420H (12 logical CPUs), approximately 16 GiB RAM, RTX 5050 Laptop with 8 GiB VRAM. Installed global torch is 2.11.0+cu128; isolated environment still required.
- Coordinator: supplied yandexresearch URL returns 404. Paper links to https://github.com/yandex-research/tabpack; verified official HEAD is `05a89e21b955f12de84889d662e15ca534019aaa`.
- Coordinator: requested user input on CPU-friendly 3-seed, 4-member budget versus a larger GPU run; no answer yet. Initial implementation must support CPU and CUDA. No paid compute.

### T37

- Authored `docs/agents/T37.md`: use `perf_counter_ns` wall time, synchronize CUDA before/after timed regions, optionally record same-stream CUDA-event diagnostics, perform no discarded training warmup, and use a fixed 10-warmup/30-measurement full-test inference protocol with explicit endpoint and method-order controls.

### T48

- Provenance contract is in `docs/agents/T48.md`; it aligns with T20 result status/schema and T35 checkpoint lineage, uses SHA-256 plus versioned canonical JSON, records local Git SHA separately from upstream reference SHA, and requires secret-free active-interpreter/software/hardware/config/data/timestamp metadata. No artifact writer or training was run.

### T49

- Wrote `docs/agents/T49.md`: PowerShell 5.1 quickstart using direct venv interpreter invocation, argument-array splatting for paths with spaces, hashed CPU/CUDA lock selection, safe preflight, explicit verified-acquisition placeholder, strict dry-run, and the three-family reduced experiment command. Read T12, T19, T28, T38, T41, and T48; corrected T28's stale `run --smoke`/unpinned examples to T41's explicit-config and injected synthetic-smoke boundary. Passed an in-memory argument round-trip probe; no install, download, source edit, CLI run, or training.

### T39

- Added `docs/agents/T39.md`; read-only Git checks confirmed unborn `master`, no local refs/remotes, and expected initial log failure. Coordinator must recover the exact requested remote identity and push authorization from the original request; the ledger's earlier empty SSH query does not configure origin or establish write access.
- Recommend stable reviewed implementation commit before measured runs, explicit-path staging and staged review, focused integration commits, and verified remote branch SHA after any authorized push. `docs/checkpoints.md` should retain attribution, exact checks/outcomes, source/run identities, limitations, and publication receipts; record a commit's own SHA in the next checkpoint/final handoff to avoid self-reference.
- Read T27 and T48 and addressed reduced-project naming and preserved execution-SHA/source inventory. Only task documentation and T39 ledger entries changed; no Git writes, source edits, installs, tests, or training.

### T44

- Wrote `docs/agents/T44.md`: golden fixtures with independent mean/sample-std oracles; tests for observation-unit boundaries, missing/failed/cancelled/in-progress/duplicate/unexpected runs, required versus optional fields, JSON/CSV round-trips and null/zero/precision handling, cross-format consistency, and permutation/filesystem/key ordering determinism. Read T20 and aligned to its manifest-declared matrix, retry acceptance, strict JSON, nulls, and sort order. No implementation tests run because source/API integration is not present yet. Directed remaining coordination questions to T20, T32, T42, and T48.

### T42

- Added `docs/agents/T42.md`: exact 56-row deterministic fixture, tiny CPU configuration for ordinary/independent/packed paths, train-only preprocessing and unknown-category checks, validation-selection/test-freeze ordering assertions, checkpoint/prediction round-trip and replay checks, and explicit `synthetic_smoke_v1` separation from real Churn results.
- Verified fixture structure with a standalone Python assertion script: 32/12/12 disjoint rows, balanced labels, numeric and categorical missingness in every split, and `violet` held out from train. No model or Churn training was run because the repository had no implementation at inspection time.

### T43

- Added `docs/agents/T43.md`: unit-test matrix for structured/actionable preflight failures (missing torch/data, low disk/RAM), safe CUDA-to-CPU fallback, deterministic thresholds, aggregation/precedence, and proof that blocked checks have no training or filesystem side effects.
- No tests run: implementation and test tree were not present; documentation-only design. Read T41's CLI contract and T42's smoke design; aligned exit/status expectations and directed implementation questions to T19/T41/T42.

### T31

- Added `docs/agents/T31.md`: proposed a two-OS Python 3.12 matrix, CPU-only/offline environment, portable module commands for Ruff/mypy/pytest, explicit `slow`/`gpu`/`data`/`smoke` marker policy, resource limits, and synthetic smoke integration aligned with T42 and T24. No workflow/source changes or tests run because package metadata and implementation are not present.

### T27

- Added `docs/agents/T27.md`: self-contained system boundary and dataflow, ordinary/homogeneous/packed model paths, explicit `[M, B, ...]` packed semantics, summed-over-members loss and probability-averaging equations, selection/resource semantics, official-vs-reduced claim boundaries, test status, and integration questions for T2/T14/T15/T21/T22.
- No tests or training run: implementation and test files were absent when documentation was drafted.

### T21

- Added `docs/agents/T21.md`: independent `nn.Linear` oracle and hand-checkable signed-logit fixture; 12 concrete homogeneous/heterogeneous forward cases covering member-specific inputs, width/depth masking, poisoned padding, explicit per-member dropout masks, parameter isolation, permutation/layout invariance, and reduced-config smoke; CPU/CUDA tolerances and mutation targets.
- Read `docs/agents/T27.md` before finishing and aligned tests with its `[M,B,...]` contract, no-sharing semantics, explicit dropout-mask requirement, and heterogeneous depth/width questions. No implementation tests were available; only the literal fixture arithmetic passed with local PyTorch 2.11.0+cu128. Directed API/mask-seam questions to T14 and optimizer/loss reuse to T22/T33.

### T26

- Wrote `docs/agents/T26.md` with a 16-case configuration validation matrix and 14-case deterministic sampling matrix, covering invalid types/ranges, finite positive log-uniform bounds, duplicate/unknown values, local RNG isolation, hash-seed independence, reproducibility, stream separation, and independent log-uniform sampling oracles.
- Read `docs/agents/T41.md` before finishing and coordinated typed config validation with its CLI schema/error contract. No automated tests were run because implementation APIs are not present; questions for T13/T34 cover seed domains, stream derivation, equal endpoints, and prefix stability.

### T8

- Pinned official Churn config uses `noisy-quantile`, global all-split binary extraction, binary-to-categorical conversion, and ordinal categorical encoding; numeric/categorical estimator fitting is train-only, but binary schema discovery inspects all split values. Official numeric NaNs are zero-filled after transformation rather than median-imputed, while the reduced protocol requires explicit train-fitted median + quantile and train-fitted mode + one-hot unknown-ignore. Documented cache-key and OOV details plus integration advice in `docs/agents/T8.md`; no tests/training run.

### T29

- Added `docs/agents/T29.md`: evidence-backed limitations for the reduced Churn-only protocol, including one-dataset external validity, four-member/three-seed reduction, omitted embeddings/Muon/pruning/tuning, optimistic-vs-conservative protocol mismatch, adaptive validation overfitting, capacity/compute and hardware confounds, preprocessing provenance, independent-reimplementation risk, and an explicit safe claim boundary. Reviewed T28 and T47 notes; no tests or training run.

### T9

- Wrote `docs/agents/T9.md`: official paper/source audit covering hybrid Muon+AdamW behavior, per-member packed state, decoupled decay, summed member losses, no schedule, and Churn optimizer ranges/configs.
- Recommended retaining the provisional float32 AdamW-only reduction for self-contained Windows CPU/CUDA portability; documented why Muon is outside scope (bfloat16 Newton–Schulz, shape scaling, hybrid groups, and pinned `muon_scale`/`muon_update_scale` mismatch) and why paper Muon-vs-AdamW results are not Churn guarantees.
- No installs, source execution, training, benchmarks, or implementation tests run. Directed T15/T22 to stock-PyTorch-AdamW equivalence checks including `grad=None` versus zero, decay, counters, padding, and checkpoint reload.

### T50

- Added `docs/agents/T50.md`: final-release acceptance contract covering tests/invariants, deterministic reproduction, data provenance and leakage, complete per-seed results, fair comparison, documentation, focused Git checkpoints, evidence packet, reviewer sign-off, and explicit reduced-protocol limitations.
- No tests or training run for this documentation-only assignment; inspected `docs/agents/T27.md` and incorporated its packed-axis, loss-reduction, probability-averaging, and validation-only selection contracts.

### T24

- Wrote `docs/agents/T24.md`: exact 12-row fixture with train medians `[40, 35]`, categorical modes `[basic, north]`, holdout numeric/category sentinels, and tests for disjoint lineage, fit-boundary row counts, train-only statistics, unknown-category all-zero blocks, stable schemas, non-mutating transforms, label isolation, cache behavior, and contamination test doubles.
- Read `docs/agents/T42.md` and `docs/agents/T48.md`; aligned with T42's train-fit spy and validation/test freeze boundary and T48's role-separated data hashes and optional preprocessing artifact identity. No implementation or dependency tests were run because the preprocessing API is not yet present.

### T47

- Added `docs/agents/T47.md`: reviewed repeated validation checkpoint/member selection overfitting; distinguished the paper's optimistic multi-seed protocol from validation-score optimism; recommended canonical untouched-test evaluation only after frozen selection, all prespecified seeds with mean/sample SD, explicit search provenance, and fresh holdout disclosure if test results influenced development.
- Read and aligned with `docs/agents/T35.md` selection/checkpoint contract. No tests or training run.

### T41

- Added `docs/agents/T41.md`: proposed `python -m tabpack.cli` contract for preflight, dry-run, run, and report; shared options; safe side-effect boundaries; canonical/noncanonical labeling; artifact/resume/report validation; stable exit codes and JSON errors; and implementation/CI acceptance cases. Read T27, T42, T43, T20, and T35 to align architecture, smoke, preflight diagnostics, aggregation, and checkpoint semantics. No source, install, training, or tests run.

### T4

- Wrote `docs/agents/T4.md` with evidence from paper §§3.5–3.8, 4.1–4.2, 5.1, Appendix C/F, pinned README/data source, and related T47 note. Defined ordinary MLP, homogeneous independent ensemble, and reduced heterogeneous packed TabPack-style methods under shared controls; distinguished paper optimistic versus conservative protocols; and documented explicit capacity, compute, tuning, validation-search, timing, three-seed, preprocessing, and non-reproduction caveats. No source, installs, training, or tests run.

### T20

- Wrote `docs/agents/T20.md`: versioned JSON result/manifest design for per-seed/per-method ROC-AUC, log-loss, timings, frozen selected snapshots, member/checkpoint identity, and secret-free environment/source/data provenance; strict completeness/duplicate/retry/hash validation; sample mean/std and paired-seed reporting; deterministic JSON/CSV exports; and acceptance cases for null/zero, malformed values, missing runs, ordering, round-trips, and timing boundaries. Read T44 and T48 and reconciled their observation-unit/statistics/testing and canonical-provenance guidance. No source implementation, dependency installation, training, measurements, or runtime tests run.

### T3

- Wrote `docs/agents/T3.md`: pinned paper/code review of online greedy selection. Official behavior is validation-score greedy without replacement, strict improvement for both candidate additions and epoch-level replacement, individual-score then stable pool-order tie breaking, and online pools containing current snapshots plus prior selected snapshots (with distinct source epochs allowed for one member). The paper's generic/default score is the task metric; official Churn reports accuracy, while this reduced protocol intentionally selects by validation ROC-AUC and must freeze selection before test evaluation. Read T35 and aligned snapshot lineage/order guidance. No source execution, installs, tests, or training run.

### T33

- Added `docs/agents/T33.md` after reading paper Sections 3.2–3.6, pinned official model/training/optimizer code, and T27's note. Confirmed mean over samples followed by sum over members; independent batch indices must index labels as well as features. Packed slices may share an allocation but must not alias trainable entries.
- Directed T14/T15/coordinator to member-local optimizer state, no cross-member clipping, explicit stopping semantics, and correct short-batch reductions; directed T21/T22/T45 to direct-gradient and pack-size-invariance checks because AdamW can conceal a wrong `/M` loss scale; coordinated RNG oracle requirements with T34. No source edits, installs, commits, tests, or training run.

### T22

- Wrote `docs/agents/T22.md`: implementation-ready packed-vs-independent AdamW cases for one and seven steps, heterogeneous per-member lr/weight decay, moments/counters, decoupled decay, zero-vs-`None` gradients, no cross-member influence, padding/inactivity boundaries, checkpoint resume, pack-size invariance, and CPU/CUDA tolerance guidance. Read and coordinated with T21, T27, T33, T35, and T45.
- Ran only a tiny CPU reference sanity check with preinstalled PyTorch `2.11.0+cu128`: first-step scalar `1.9880000002`, zero-gradient decay `1.998`, and `None`-gradient skip `2.0`/absent state; follow-up zero-gradient state counter reached 2. Packed equivalence tests were not run because implementation/API files are not yet available. No source edits, installs, commits, or training.

### T18

- Added `docs/agents/T18.md`: approved immutable Churn source-lock and safe NPY/index validation, preserved canonical split order, train-only median/quantile and mode/one-hot pipeline, frozen-state API/fingerprint, fail-closed typed errors, test-finalization boundary, and nine acceptance-test groups. Read T24/T48/T42/T35 and addressed lineage, provenance, smoke injection, and checkpoint coordination.
- Read-only source audit found upstream binary extraction scans all splits; reduced design avoids it. Hugging Face metadata reports data revision `dfb85cb493bfb7493de0b8c0bd9e2f9e6a56baa1`, archive size 189856645, and published LFS SHA-256 `89338c628fed24af03084c9348ca0a5c8ca4f12f8b988a576d9d1711cc661558`; actual Churn member evidence remains directed to T7. No archive download, runtime tests, source edits, installs, commits, or training performed.

### T19

- Wrote `docs/agents/T19.md`: safe offline preflight command/checklist for active Python/dependencies, PyTorch/CUDA and explicit-device policy, RAM/disk estimation, canonical Churn data/schema and train-only preprocessing, reproducibility/provenance fields, stable diagnostics/exit statuses, and actionable failure recovery. Read T12, T18, T41, T43, and T48 and aligned lock, data, CLI, test, and secret-free provenance contracts. Captured bounded local probes only; no installs, downloads, large allocations, source changes, commits, or training.

### T17

- Wrote `docs/agents/T17.md`: standard-library grouped ROC-AUC with half-credit ties and explicit single-class rejection; stable clipped log-loss accepting single-class labels; proposed float64/`2**-52` evaluation policy, strict input validation, API/formulas/test vectors, and probability-averaging guidance.
- Passed 8,604 exhaustive small AUC comparisons against an independent pairwise oracle, five named AUC cases, four log-loss cases, and two single-class AUC rejection checks. Read T20 and aligned its null-plus-diagnostic reporting boundary with metric exceptions. Documentation only; no source/install/commit/training changes.

### T38

- Added `docs/agents/T38.md`: require verified HTTPS on every redirect, immutable source lock, SHA-256/size verification, bounded decompression/member handling, exact Churn allowlists, fresh staging, Windows path/link rejection, atomic publication, safe NPY loading, and hard failure rather than IBM Telco/synthetic/stale-data substitution.
- Independently queried the same Hugging Face revision/size/LFS digest as T18. Reading only an archive prefix or Range response cannot verify the full archive SHA-256; partial Churn acquisition needs separately pinned member hashes and an explicit unverified-archive receipt. Read T18/T24/T28/T35/T48 and addressed source-lock, verified-byte handoff, and provenance questions. T7 must still confirm member inventory and concrete extraction ceilings. No archive download, source edit, install, commit, training, or runtime tests performed.

### T51

- Independently authored `tabpack/models.py`, `tabpack/optim.py`, `tests/test_models.py`, `tests/test_optim.py`, and `docs/agents/T51.md`. OrdinaryMLP and depth-indexed rectangular PackedMLP preserve `[M,B,F]` semantics, heterogeneous widths/depths/dropout, structural masking, compact member snapshots, sample-mean/member-sum BCE, and logical/stored parameter accounting. PackedAdamW has member-vector hyperparameters, per-parameter/member moments/counters, masked padding, pause/resume, and model-aware zero-bias-decay groups.
- Read and coordinated with T14/T15/T21/T22/T33/T36/T46 (plus T27 introduction); task note documents exact constructors, tensor names, dropout-mask seam, optimizer groups/factory, snapshot helpers, and integration boundaries.
- Last completed focused command `python -B -m pytest tests/test_models.py tests/test_optim.py -p no:cacheprovider -q`: **78 passed in 4.77s**, including **2 CUDA tests**, with one stock-PyTorch sparse-fixture warning. Independent Linear and one/seven-step AdamW oracles cover logits, raw gradients, parameters, moments, counters, dirty padding, loss scaling, pack-size invariance, distinct member batches, zero/None gradients, and state round trips.
- Later warning-cleanup rechecks timed out; cleanup was reverted. An import-only probe printed its start marker then stalled at `import torch` before pytest/project imports. Final source/tests match the completed passing run; coordinator should recheck torch import in the chosen isolated environment. No installs, downloads, Git writes, other-source changes, or experiment training.

### T53

- Implemented `tabpack/core.py`, `tests/test_core.py`, and `docs/agents/T53.md`: immutable validated ReducedConfig/member records, prefix-stable named SHA-256 seed streams and local field sampling, tie-aware AUC/log-loss, compensated probability averaging, strict best-singleton greedy selection with individual-AUC/pool-order ties, strict online incumbent retention, deterministic complete/incomplete result summaries, paired-seed deltas, and strict canonical JSON.
- Reviewed T3/T13/T16/T17/T20/T23/T26/T32/T44 and T51 model constructors; T25 was unavailable. Followed decision 9 rather than T16's empty-baseline/canonical-sort/warm-start proposal. Exact APIs, model-stream integration, snapshot freezing, and aggregation manifest/provenance boundary are documented in the task note.
- Final focused command `python -m pytest tests/test_core.py -q -p no:cacheprovider`: **140 passed in 6.60s**, including 8,604 exhaustive small AUC comparisons against a pairwise oracle, T23 exact selector fixtures, subprocess hash-seed reproducibility, and aggregation completeness/statistics/ordering tests. Used installed Python/pytest; no venv or Ruff was available. No installs, Git writes, dataset downloads, model training, or long runs.

### T52

- Implemented `tabpack/data.py`, `tests/test_data.py`, `docs/data-source.json`, and `docs/agents/T52.md`: fixed immutable archive URL/SHA-256/size, HTTPS redirect allowlist and bounded verified downloads, private-snapshot archive verification, bounded Windows-safe tar allowlisting, atomic Churn-only publication, strict offline member-byte/schema/split validation, preserved read-only lineage, train-only median/normal-quantile plus mode/one-hot preprocessing, complete learned-state fingerprint, and explicit `synthetic_smoke_v1` test seam.
- Read T8/T18/T24/T38/T42; T1/T7/T30 notes remained unavailable. Member hashes are derived only from the fully verified retained canonical archive on each load, never from an editable self-certifying receipt or fabricated static inventory. Bounded HEAD evidence confirmed current delivery host `us.aws.cdn.hf.co`; no archive was downloaded and no actual Churn member inventory or real-data success is claimed. Public acquisition/load/preparation APIs and integration requirements are in the task note.
- Final focused command `python -m pytest tests/test_data.py -q`: **70 passed, 9 skipped in 0.76s**. Eight train-only preprocessing/oracle tests await scikit-learn; one filesystem symlink test requires Windows privilege. Tar-link, sparse/PAX/resource-limit, fail-closed loader, publication, safe-NPY, split-integrity, and provenance tests passed. Ruff was unavailable. No installs, Git writes, other-source edits, model training, or long runs.

### T55

- Added `README.md`, `docs/protocol.md`, and `.github/workflows/tests.yml`; documented direct-venv PowerShell/Linux setup, explicit CPU/cu128 PyTorch 2.11.0 installation, the six coordinator CLI commands, current source-module responsibilities, train-only preprocessing, local-only inference artifacts, no exact resume, and no measured Churn results.
- The workflow is a Windows/Linux Python 3.12 CPU matrix. It installs `torch==2.11.0` from the CPU index, then `requirements/base.txt`, runs `pip check`, `pytest -m 'not gpu'`, and `ruff check tabpack tests`; it does not download data or run experiments. No hosted workflow was run.
- Read `docs/checkpoints.md`, T51/T52/T53, and T4/T29/T31/T32/T49 plus current source. `tabpack/experiment.py` and `tabpack/cli.py` were absent and `reports/` was empty, so docs explicitly label the agreed command surface integration-pending. No source/install/data/training/Git operation was performed.
- Read-only isolated lint command `& ".\\.venv\\Scripts\\python.exe" -m ruff check tabpack tests --no-cache` found one existing source issue: `tabpack/core.py:9` import-order `I001`; no `--fix` was used. Coordinator must fix and rerun CI checks. Integration caveats and exact patience-comparator question are in `docs/agents/T55.md`.

### T54

- Reconciled every original T1–T50 assignment: **50/50 notes complete**, with 42 existing notes and eight recovered under their original filenames (`T1`, `T5`, `T7`, `T10`, `T25`, `T30`, `T34`, `T40`). Recovered notes explicitly state that T54 recovered them after an interrupted callback; this is ledger reconciliation, not a claim that the original callbacks returned.
- Corrected misleading implementation/test task titles to design/review where those agents wrote documentation only. Preserved actual receipts: T51 78 focused tests including 2 CUDA; T52 70 passed/9 skipped; T53 140 passed including 8,604 AUC oracle vectors. No new source, tests, install, archive download, training or Git operation was performed by T54.
- Reconciled official reference SHA `05a89e21b955f12de84889d662e15ca534019aaa` as valid and distinct from the unborn local SHA. Official Churn uses accuracy as `score`; local reduced selection/reporting uses validation/test ROC-AUC by deliberate protocol change. The HF archive identity is provider-published (size 189856645, SHA-256 `89338c628fed24af03084c9348ca0a5c8ca4f12f8b988a576d9d1711cc661558`) and has not been locally downloaded or parsed; T52's full-archive verification is the current trust boundary. Original dataset licensing, actual member inventory, semantic target meaning and end-to-end Churn measurements remain blockers.

### T57

- Added `tests/test_experiment.py` and `tests/test_cli.py` plus `docs/agents/T57.md`. The tests use only `synthetic_smoke_v1`, a one-seed/two-member/two-epoch CPU config with widths 4/8, and cover all three methods, finite metrics, validation-freeze ordering spies, weights-only checkpoint replay, deterministic two-run replay, non-empty output refusal, CLI smoke/report, and incomplete/matrix-tamper report rejection.
- Required command `& ".\\.venv\\Scripts\\python.exe" -m pytest tests/test_experiment.py tests/test_cli.py -q`: **6 passed, 1 failed in 6.98s**. The failing assertion intentionally exposes that `report_comparison()` accepts a tampered manifest `comparison_hash`; matrix tampering and all other checks pass. Exact gap and no-silent-skip rationale are in `docs/agents/T57.md`.

### T56

- Wrote `docs/agents/T56.md` after a read-only review of `tabpack/experiment.py`, `tabpack/cli.py`, the checked-in Churn report, and protocol/provenance claims. New blockers beyond the coordinator's listed RNG/epoch/config/determinism/report-hash fixes: `ReducedConfig.dtype` is silently ignored; reported `fit_wall_s` excludes construction and has fixed method-order fairness limitations; interrupted runs lack finalized cancelled/progress records; report generation does not verify result/checkpoint integrity; checkpoint entries lack explicit member/epoch lineage; and source identity is absent from the comparison hash. The checked-in measurements are pre-commit evidence (`source.local_commit: null`) and require a committed corrected rerun before final release.
- Read T51/T53 and documented the positive independent-snapshot path, caller-owned freeze/report boundaries, and exact integration fixes. No source/test edits, installs, downloads, training, or tests were performed by T56. T57's note was read; its 6-pass/1-intentional-failure result independently confirms the manifest `comparison_hash` tamper gap.

## Decisions

1. Each agent MUST read this file first; set only its row to running, then done/blocked with a one-line result. Append findings under its own `### Tn` heading. Use small apply_patch hunks; reread after any concurrent-edit conflict. Do not rewrite the whole ledger.
2. Each agent writes a concise `docs/agents/Tn.md` task note: assignment, sources/evidence, findings, tests run (or not run), integration advice, and coordination. Read at least one related agent's available note/findings before finishing and address it, or leave an explicit directed question in your own note. This is the shared communication channel; do not claim messages that were not read.
3. No agent may commit, stage, install dependencies, change Git settings/remotes, run long training, or modify another task's files. The coordinator serializes meaningful commits after review, records attribution and tests in `docs/checkpoints.md`, and owns packaging, CLI, training, downloading, integration, and final README. No nested agents (50 already requested).
4. All source is independently authored, not vendored from upstream. References are pinned to the SHA above; do not execute/import official code. Official reference may be read in ignored `.reference/tabpack` once available, or raw GitHub pinned URLs. Cite paper v1 https://arxiv.org/html/2607.05380v1. Avoid copying private local config/logs into repo.
5. Research/review assignments modify only their task note and their own ledger row/findings. Implementation assignments own only their specified module and note. Tests own only their specified test file. Documentation assignments own only their named document. Never pretend a test passed if dependencies/code are not ready.
6. Reduced protocol (provisional, exposed via config): 3 seeds [0,1,2], 4 base members, maximum selected ensemble size 4, max 100 epochs, patience 16, batch 256, float32. Baseline: depth 2, width 64, dropout .1, AdamW lr .001, wd .0001. Heterogeneous: depths [1,2,3], widths [32,64,96], dropout [0,.1,.2], log-uniform lr [0.0003,0.003], wd [0.000001,0.001]. No periodic feature embeddings, no Muon, no base pruning, no tuning. Report as reduced TabPack-style optimistic evaluation, not paper-equivalent scores. Reassess after research/user input.
7. Preserve canonical official Churn train/val/test splits if obtainable. Numeric median imputation + quantile-normal transform fit ONLY on train; categorical most-frequent imputation + one-hot encoding with unknown categories ignored fit ONLY on train. Never silently substitute IBM Telco or synthetic data. Synthetic data is test-only.
8. Core behavior: no parameter sharing; pack dimension first; per-member loss averaged over samples and SUMMED over members; independent training permutations; probability averaging; validation ROC-AUC for checkpoint/greedy choices; final test evaluation only after freezing selection. Online candidate pool = current epoch members plus previous selected checkpoint snapshots; greedy without replacement and strictly improving; snapshots may share member ID but have different epochs.
9. Integration overrides: best singleton ALWAYS selected (T3/T23), no T16 empty-ensemble baseline or warm-started selection. Homogeneous ensemble checkpointed jointly by all-member mean validation AUC. Biases have zero decay everywhere. Final weights-only inference checkpoints, not resumable training. Implementation owners T51/T52/T53 follow exact APIs from their prompts. Coordinator runs installs/tests and final training; agents may run short tests with .venv once available but no installs/Git writes. Initial unfinished task notes are recovered explicitly, not assumed complete.
