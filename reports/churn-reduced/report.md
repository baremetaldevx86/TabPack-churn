# Reduced Churn comparison

Protocol: `reduced-churn-v1` (reduced optimistic workflow)
Comparison hash: `b62a7db2991c06d4e218e8feaf3d8ec32f317a47ac97b193ef30734ace4f969d`
Seeds: `[0, 1, 2]`; device: `cuda`

Local test metrics are ROC-AUC/log-loss. The official paper's Churn table reports accuracy; these are not numerically comparable.

| Method | Test ROC-AUC mean ± sample SD | Test log-loss mean ± sample SD | Fit seconds mean ± sample SD |
| --- | ---: | ---: | ---: |
| `mlp` | 0.855624 ± 0.001718 | 0.346776 ± 0.002816 | 15.134903 ± 4.829494 |
| `independent_mlp_ensemble` | 0.856576 ± 0.000944 | 0.344509 ± 0.001083 | 65.673245 ± 5.275698 |
| `packed_heterogeneous_ensemble` | 0.858702 ± 0.002105 | 0.343824 ± 0.004010 | 84.258271 ± 42.193035 |

Every seed result is retained in `per_seed.csv` and under the local run directory; validation selection is frozen before test evaluation.
