# SANDARBH: Final Research Report

## 1. Plain-language abstract

SANDARBH studies whether conversational context, soft annotation targets, probability calibration, and confidence-based abstention change binary detection behaviour on AUTALIC. All decisions were frozen before a single locked-test evaluation. The test contained 341 examples, including 38 positive-labelled examples. Evidence was mixed: context changed the balance between accuracy and positive recall; soft-target training restored recall but worsened several probability-quality measures relative to contextual hard-label training; calibration helped E6 and E7 more consistently than E5; abstention reduced observed error while also excluding positive-labelled examples. These are research findings, not a deployment or safety guarantee.

## 2. Motivation

Target-only classification can miss surrounding context, while contextual models may change false-positive and false-negative patterns. This project evaluates those trade-offs under a leakage-controlled, predeclared protocol.

## 3. Research questions

1. Does conversational context help compared with target-only input?
2. How does soft-vote-fraction training compare with hard-label training under identical contextual input construction?
3. How do frozen temperature calibration and confidence-based abstention affect probability quality, coverage, and risk?

## 4. Dataset description

The frozen modeling set contains 2278 eligible examples. The locked test contains 341 examples: 303 label 0 and 38 label 1. Source sentences are never reproduced in this report.

## 5. Annotation interpretation

Class 1 means a majority ableist annotation under this benchmark mapping. Class 0 means no majority positive annotation; it does not establish that language is safe or non-ableist. Soft targets are annotator vote fractions, not ground-truth probabilities.

## 6. Eligibility and exclusions

Eligibility and exclusions were fixed during the audit. Excluded rows never entered modeling partitions, and Phase 7 does not revise those decisions.

## 7. Overlap grouping and leakage control

Exact normalized overlap links were consolidated into Rule B groups and kept intact across partitions. This controls detected exact overlap but cannot detect every paraphrase or semantic near-duplicate.

## 8. Frozen partition design

Train, development-tuning, calibration, and test partitions were frozen before model fitting. The test contained 213 Rule B groups, with no group crossing partitions.

## 9. Classical baselines

E1/E2 are saved TF-IDF logistic pipelines; E3/E4 are saved TF-IDF LinearSVC pipelines. E1/E3 use target text and E2/E4 use context. DUMMY always predicts class 0. No baseline was retrained or refitted.

## 10. Transformer architecture

E5–E7 use frozen DistilRoBERTa seed-42 checkpoints. E5 is target-only hard-label training; E6 is contextual hard-label training; E7 is contextual soft-target training. Inference used CPU FP32.

## 11. Target-only versus contextual inputs

Classical context comparisons were mixed: E2−E1 changed macro-F1 by 0.0770, and E4−E3 by 0.0300. For transformers, E6−E5 changed macro-F1 by 0.0172 but positive recall by -0.2632; the recall interval was [-0.5000, -0.0833]. Context therefore did not yield uniform improvement across metrics.

## 12. Hard versus soft targets

E7−E6 changed macro-F1 by -0.0318, positive F1 by 0.0054, and positive recall by 0.2632. The positive-recall interval was [0.1304, 0.4286]. E7 improved ranking metrics but had worse NLL, Brier, ECE, soft cross-entropy, and soft Brier than E6. This is a trade-off, not an unconditional winner.

## 13. Calibration methodology

Scalar temperatures were fitted only on the 171-example calibration partition, which had 19 positives, then frozen. Test logits were divided by the frozen temperature before softmax; every argmax was preserved.

## 14. Selective-prediction methodology

The primary evaluation uses complete coverage. The secondary policy accepts predictions with maximum calibrated probability at least 0.60. Thresholds 0.50–0.99 are descriptive and were frozen before test access.

## 15. Frozen final evaluation policy

The authoritative policy hash is `b5765e9431a4bf01e8abfeeec1762061b1c0a26eec4ea26c326abb7245ac7541`. It fixes models, temperatures, thresholds, metrics, comparisons, bootstrap settings, and interpretation constraints.

## 16. Locked-test classification results

| System | Macro-F1 | Pos P | Pos R | Pos F1 | Accuracy | Balanced accuracy | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DUMMY | 0.4705 | 0.0000 | 0.0000 | 0.0000 | 0.8886 | 0.5000 | NA | NA |
| E1 | 0.5769 | 0.3043 | 0.1842 | 0.2295 | 0.8622 | 0.5657 | 0.7005 | 0.2698 |
| E2 | 0.6539 | 0.5882 | 0.2632 | 0.3636 | 0.8974 | 0.6200 | 0.7343 | 0.4053 |
| E3 | 0.5991 | 0.3030 | 0.2632 | 0.2817 | 0.8504 | 0.5936 | 0.7064 | 0.3381 |
| E4 | 0.6291 | 0.6154 | 0.2105 | 0.3137 | 0.8974 | 0.5970 | 0.7405 | 0.3843 |
| E5 | 0.6184 | 0.2700 | 0.7105 | 0.3913 | 0.7537 | 0.7348 | 0.7657 | 0.2707 |
| E6 | 0.6356 | 0.3148 | 0.4474 | 0.3696 | 0.8299 | 0.6626 | 0.7591 | 0.3477 |
| E7 | 0.6039 | 0.2547 | 0.7105 | 0.3750 | 0.7361 | 0.7249 | 0.7862 | 0.4495 |

## 17. Probability and calibration results

| System | State | NLL | Brier | ECE | Soft Brier |
|---|---:|---:|---:|---:|---:|
| E1 | native_uncalibrated | 0.3566 | 0.1052 | NA | NA |
| E2 | native_uncalibrated | 0.6401 | 0.2235 | NA | NA |
| E5 | uncalibrated | 0.5059 | 0.1694 | 0.2338 | 0.1177 |
| E5 | calibrated | 0.5056 | 0.1691 | 0.2357 | 0.1173 |
| E6 | uncalibrated | 0.4371 | 0.1337 | 0.2076 | 0.0816 |
| E6 | calibrated | 0.3864 | 0.1197 | 0.1362 | 0.0723 |
| E7 | uncalibrated | 0.5808 | 0.1952 | 0.3320 | 0.1382 |
| E7 | calibrated | 0.5640 | 0.1856 | 0.2996 | 0.1314 |

## 18. Selective prediction at 0.60

| System | Accepted | Coverage | Risk | Positive coverage | Negative coverage |
|---|---:|---:|---:|---:|---:|
| E5 | 289 | 0.8475 | 0.2249 | 0.7895 | 0.8548 |
| E6 | 305 | 0.8944 | 0.1311 | 0.8684 | 0.8977 |
| E7 | 279 | 0.8182 | 0.2151 | 0.7895 | 0.8218 |

Lower observed selective risk accompanies reduced coverage and must not be interpreted as safety.

## 19. Bootstrap uncertainty

Intervals use 5,000 paired complete-group bootstrap replicates with seed 20261006. They summarize uncertainty for this sample and are not p-values or universal proof.

## 20. RQ1 conclusion — context

Observed evidence is mixed. Classical contextual systems increased macro-F1 relative to their target-only counterparts. E6 slightly increased macro-F1 over E5 but reduced positive recall substantially. Different metrics support different interpretations.

## 21. RQ2 conclusion — soft annotation targets

E7 increased positive recall and ranking quality relative to E6, while lowering accuracy and macro-F1 and worsening multiple calibrated probability-quality metrics. Soft vote fractions must not be treated as ground-truth probabilities.

## 22. RQ3 conclusion — calibration and abstention

Calibration improved E6 and E7 NLL, Brier, and ECE. E5 had small NLL/Brier improvements but slightly worse ECE. At 0.60, E5/E6/E7 coverage was 0.8475/0.8944/0.8182, with positive-label coverage 0.7895/0.8684/0.7895. Evidence is mixed and does not establish deployment reliability.

## 23. Error, risk, ethics, and prohibited uses

Errors can be harmful in either direction. The system is not diagnostic, clinical, moderation-ready, or suitable for automated punishment. A negative prediction does not make language safe. Human review and broader validation would be essential for any consequential use. Individual test examples were not inspected or exposed.

## 24. Reproducibility and integrity

The test was accessed once under an immutable receipt. Predictions are text-free. The independent Phase 6 validator passed 42/42 checks. No retraining, recalibration, threshold tuning, seed shopping, p-values, post-test winner selection, or policy change occurred.

## 25. Limitations

The test has only 38 positives. Results are limited to AUTALIC and this frozen split. Exact-overlap grouping misses some paraphrases. Calibration used 171 examples with 19 positives. Bootstrap intervals depend on the observed groups. Selective prediction can reject positive-labelled examples. The UI is a local research demonstration and may produce incorrect or harmful predictions.

## 26. Conclusion and future work

SANDARBH provides a reproducible locked evaluation of context, soft targets, calibration, and abstention. Future work should use new external data, broader overlap detection, larger positive-class samples, independent annotation studies, and separately pre-registered evaluations. The locked test must not be retuned or rerun.
