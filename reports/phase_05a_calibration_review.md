# Phase 05A Calibration and Selective-Prediction Review

**Run status:** `COMPLETED`  
**Policy status:** `PENDING_RESEARCHER_REVIEW`  
**Generated (UTC):** 2026-10-06T04:29:40.443315Z  

Post-calibration changes below are apparent fit-set diagnostics because fitting and measurement both use the 171-example `dev_calibration` set. No model or policy is selected, and the test partition was not accessed.

## Calibration diagnostics

| Experiment | Version | Temperature | NLL | Brier | ECE | ROC-AUC | AP | Soft CE | Soft Brier | Soft MAD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E5 | uncalibrated | 1.000000 | 0.482195 | 0.156725 | 0.170204 | 0.634349 | 0.186541 | 0.499196 | 0.099907 | 0.243935 |
| E5 | temperature_scaled | 1.023717 | 0.482117 | 0.156668 | 0.173142 | 0.634349 | 0.186541 | 0.498723 | 0.099740 | 0.245512 |
| E6 | uncalibrated | 1.000000 | 0.430738 | 0.129932 | 0.167469 | 0.589681 | 0.229672 | 0.458256 | 0.078699 | 0.243502 |
| E6 | temperature_scaled | 0.602057 | 0.396391 | 0.117948 | 0.091194 | 0.589681 | 0.229672 | 0.442098 | 0.072140 | 0.203621 |
| E7 | uncalibrated | 1.000000 | 0.571820 | 0.190557 | 0.291248 | 0.630886 | 0.222719 | 0.573430 | 0.127644 | 0.320666 |
| E7 | temperature_scaled | 0.615406 | 0.554325 | 0.180282 | 0.241176 | 0.630886 | 0.222719 | 0.556942 | 0.117782 | 0.288627 |

## Risk/coverage checkpoints

| Experiment | Version | Threshold | Coverage | Risk | Positive coverage | Accepted |
|---|---|---:|---:|---:|---:|---:|
| E5 | uncalibrated | 0.50 | 1.000000 | 0.198830 | 1.000000 | 171 |
| E5 | uncalibrated | 0.70 | 0.742690 | 0.173228 | 0.684211 | 127 |
| E5 | uncalibrated | 0.80 | 0.578947 | 0.141414 | 0.421053 | 99 |
| E5 | uncalibrated | 0.90 | 0.175439 | 0.066667 | 0.052632 | 30 |
| E5 | temperature_scaled | 0.50 | 1.000000 | 0.198830 | 1.000000 | 171 |
| E5 | temperature_scaled | 0.70 | 0.725146 | 0.169355 | 0.684211 | 124 |
| E5 | temperature_scaled | 0.80 | 0.561404 | 0.125000 | 0.368421 | 96 |
| E5 | temperature_scaled | 0.90 | 0.134503 | 0.000000 | 0.000000 | 23 |
| E6 | uncalibrated | 0.50 | 1.000000 | 0.140351 | 1.000000 | 171 |
| E6 | uncalibrated | 0.70 | 0.736842 | 0.095238 | 0.578947 | 126 |
| E6 | uncalibrated | 0.80 | 0.362573 | 0.096774 | 0.315789 | 62 |
| E6 | uncalibrated | 0.90 | 0.000000 |  | 0.000000 | 0 |
| E6 | temperature_scaled | 0.50 | 1.000000 | 0.140351 | 1.000000 | 171 |
| E6 | temperature_scaled | 0.70 | 0.836257 | 0.104895 | 0.736842 | 143 |
| E6 | temperature_scaled | 0.80 | 0.736842 | 0.095238 | 0.578947 | 126 |
| E6 | temperature_scaled | 0.90 | 0.450292 | 0.116883 | 0.421053 | 77 |
| E7 | uncalibrated | 0.50 | 1.000000 | 0.263158 | 1.000000 | 171 |
| E7 | uncalibrated | 0.70 | 0.368421 | 0.174603 | 0.368421 | 63 |
| E7 | uncalibrated | 0.80 | 0.017544 | 0.666667 | 0.052632 | 3 |
| E7 | uncalibrated | 0.90 | 0.000000 |  | 0.000000 | 0 |
| E7 | temperature_scaled | 0.50 | 1.000000 | 0.263158 | 1.000000 | 171 |
| E7 | temperature_scaled | 0.70 | 0.631579 | 0.175926 | 0.473684 | 108 |
| E7 | temperature_scaled | 0.80 | 0.362573 | 0.177419 | 0.368421 | 62 |
| E7 | temperature_scaled | 0.90 | 0.017544 | 0.666667 | 0.052632 | 3 |

## Interpretation

Temperature scaling preserved every native argmax prediction and associated classification metric. Reliability and threshold results are unstable with only 19 hard-positive examples; abstention can disproportionately remove positive-labelled examples. Final validity requires later locked-test evaluation after researcher policy approval.
