# Phase 04B DistilRoBERTa Development Results

**Status:** `COMPLETED`  
**Generated (UTC):** 2026-10-05T16:48:43.146571Z  

These are `dev_tune` development results, not final test performance. Seed 42 was predesignated for later calibration/final stages and was not selected by performance. No `dev_calibration` or test inference, calibration, threshold tuning, or selective prediction occurred.

## Selected checkpoints

| Experiment | Seed | Epoch | Macro-F1 | Positive P/R/F1 | Accuracy | Balanced accuracy | AP | ROC-AUC | NLL | Brier | ECE |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| E5 | 42 | 3 | 0.514894 | 0.146341/0.315789/0.200000 | 0.719298 | 0.542763 | 0.272174 | 0.623615 | 0.500445 | 0.168168 | 0.187051 |
| E5 | 43 | 3 | 0.535106 | 0.170732/0.368421/0.233333 | 0.730994 | 0.572368 | 0.193411 | 0.624307 | 0.512479 | 0.171484 | 0.213217 |
| E5 | 44 | 3 | 0.598169 | 0.272727/0.315789/0.292683 | 0.830409 | 0.605263 | 0.332864 | 0.658934 | 0.422082 | 0.132683 | 0.149818 |
| E6 | 42 | 2 | 0.592055 | 0.333333/0.210526/0.258065 | 0.865497 | 0.578947 | 0.204599 | 0.573407 | 0.431517 | 0.129966 | 0.163981 |
| E6 | 43 | 3 | 0.561095 | 0.208333/0.263158/0.232558 | 0.807018 | 0.569079 | 0.222116 | 0.672784 | 0.446764 | 0.142994 | 0.157495 |
| E6 | 44 | 3 | 0.601781 | 0.258065/0.421053/0.320000 | 0.801170 | 0.634868 | 0.234832 | 0.725069 | 0.427689 | 0.140397 | 0.129165 |
| E7 | 42 | 2 | 0.609258 | 0.264706/0.473684/0.339623 | 0.795322 | 0.654605 | 0.251145 | 0.671053 | 0.557257 | 0.184483 | 0.296079 |
| E7 | 43 | 3 | 0.526316 | 0.175439/0.526316/0.263158 | 0.672515 | 0.608553 | 0.228066 | 0.708449 | 0.577902 | 0.196824 | 0.295951 |
| E7 | 44 | 2 | 0.520016 | 0.189189/0.736842/0.301075 | 0.619883 | 0.671053 | 0.237736 | 0.733726 | 0.662195 | 0.234087 | 0.352990 |

## Across-seed macro-F1

| Experiment | Mean | Sample SD | Min | Max |
|---|---:|---:|---:|---:|
| E5 | 0.549390 | 0.043436 | 0.514894 | 0.598169 |
| E6 | 0.584977 | 0.021246 | 0.561095 | 0.601781 |
| E7 | 0.551863 | 0.049805 | 0.520016 | 0.609258 |

## Interpretation

E5-versus-E6 and E6-versus-E7 paired differences are stored in `comparisons.json`. They are descriptive across the three fixed seeds; no significance test or overall winner is declared. ECE and all development differences are unstable with only 19 hard-positive dev_tune examples. Soft-target agreement is not described as ground-truth probability calibration.

Only selected seed-42 checkpoints are retained locally. Seeds 43 and 44 retain metrics and predictions but no weights. No final generalization claim is permitted.
