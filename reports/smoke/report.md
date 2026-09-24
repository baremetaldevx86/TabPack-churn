# Reduced Churn comparison

Protocol: `reduced-churn-v1` (reduced optimistic workflow)
Comparison hash: `a6ff0578f55d63a80f89849b079830a39f16d675d968c73ef4048425a92292d7`
Seeds: `[0]`; device: `cpu`

Local test metrics are ROC-AUC/log-loss. The official paper's Churn table reports accuracy; these are not numerically comparable.

| Method | Test ROC-AUC mean ± sample SD | Test log-loss mean ± sample SD | Fit seconds mean ± sample SD |
| --- | ---: | ---: | ---: |
| `mlp` | 1.000000 | 0.581512 | 0.230579 |
| `independent_mlp_ensemble` | 1.000000 | 0.629794 | 0.555294 |
| `packed_heterogeneous_ensemble` | 0.611111 | 0.669124 | 0.375278 |

Every seed result is retained under the run directory; validation selection is frozen before test evaluation.
