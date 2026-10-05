# Phase 03 Classical Baselines

**Status:** `COMPLETED`  
**Generated (UTC):** 2026-10-05T04:35:24.963319Z  

Four TF-IDF baselines were fitted on `train` only and selected independently on `dev_tune`. No model-facing access to `dev_calibration` or `test` occurred. These are development results, not final generalization estimates.

## Selected candidates

| Experiment | Input | Classifier | C | Class weight | Macro-F1 | Positive F1 | Average precision | ROC-AUC | Vocabulary |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| E1 | target | logistic_regression | 10.0 | balanced | 0.556581 | 0.193548 | 0.210746 | 0.616343 | 6201 |
| E2 | context | logistic_regression | 0.1 | balanced | 0.518024 | 0.095238 | 0.192115 | 0.589681 | 21258 |
| E3 | target | linear_svc | 0.1 | balanced | 0.570312 | 0.228571 | 0.224783 | 0.631579 | 6201 |
| E4 | context | linear_svc | 0.1 | balanced | 0.514205 | 0.090909 | 0.200341 | 0.591413 | 21258 |

Selection used exact unrounded macro-F1 over labels `[0, 1]`, then smaller C, then null class weight before `balanced`. No decision threshold was tuned. Logistic Regression probabilities are native but uncalibrated. LinearSVC scores are margins, not probabilities.

## Dummy benchmark

The train-only majority class was `0`. Its dev_tune macro-F1 was 0.470588. This is a descriptive benchmark, not a competitive model selected for deployment.

## Limitations

- `dev_tune` contains only 19 hard-positive examples; differences may be unstable.
- Concatenated context does not explicitly encode preceding/target/following roles.
- Serialized TF-IDF pipelines contain learned vocabulary terms and are kept local; reports and tabular outputs contain no raw sentences or feature lists.
- No overall project winner was selected and no experiment was dropped.
- Calibration, selective prediction, transformers, threshold selection, and final test evaluation were not performed.
