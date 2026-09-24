# Reduced Churn comparison

Protocol: `reduced-churn-v1` (reduced optimistic workflow)
Comparison hash: `87a5cb90ce7f30324d64dad53e22b2cc7cad1205deaab7e5f53cc847d73873d7`
Seeds: `[0, 1, 2]`; device: `cuda`

Local test metrics are ROC-AUC/log-loss. The official paper's Churn table reports accuracy; these are not numerically comparable.

| Method | Test ROC-AUC mean ± sample SD | Test log-loss mean ± sample SD | Fit seconds mean ± sample SD |
| --- | ---: | ---: | ---: |
| `mlp` | 0.855414 ± 0.002412 | 0.345809 ± 0.003415 | 9.630956 ± 2.041289 |
| `independent_mlp_ensemble` | 0.856576 ± 0.000944 | 0.344509 ± 0.001083 | 50.201032 ± 5.714166 |
| `packed_heterogeneous_ensemble` | 0.858825 ± 0.002562 | 0.343488 ± 0.004115 | 60.960035 ± 19.556873 |

Every seed result is retained in `per_seed.csv` and under the local run directory; validation selection is frozen before test evaluation.
