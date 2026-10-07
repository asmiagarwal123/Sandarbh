# SANDARBH Phase 6 Final Test Results

One-time locked-test evaluation under successor policy `b5765e9431a4bf01e8abfeeec1762061b1c0a26eec4ea26c326abb7245ac7541`. No model winner was selected.

## Cohort

341 examples: 38 positive-labelled, 303 other; 213 Rule B groups.

## Classification

| System | Macro-F1 | Positive F1 | Positive recall | Balanced accuracy | Accuracy | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 | 0.576912 | 0.229508 | 0.184211 | 0.565703 | 0.862170 | 0.700538 | 0.269825 |
| E2 | 0.653907 | 0.363636 | 0.263158 | 0.620028 | 0.897361 | 0.734323 | 0.405286 |
| E3 | 0.599110 | 0.281690 | 0.263158 | 0.593625 | 0.850440 | 0.706444 | 0.338061 |
| E4 | 0.629129 | 0.313725 | 0.210526 | 0.597012 | 0.897361 | 0.740490 | 0.384330 |
| E5 | 0.618446 | 0.391304 | 0.710526 | 0.734801 | 0.753666 | 0.765677 | 0.270712 |
| E6 | 0.635630 | 0.369565 | 0.447368 | 0.662628 | 0.829912 | 0.759076 | 0.347673 |
| E7 | 0.603857 | 0.375000 | 0.710526 | 0.724900 | 0.736070 | 0.786173 | 0.449519 |
| DUMMY | 0.470497 | 0.000000 | 0.000000 | 0.500000 | 0.888563 | NA | NA |

## Frozen comparisons

| Direction | Metric | Difference |
|---|---|---:|
| E6 minus E5 | macro_f1 | 0.017184 |
| E6 minus E5 | positive_precision | 0.044815 |
| E6 minus E5 | positive_recall | -0.263158 |
| E6 minus E5 | positive_f1 | -0.021739 |
| E6 minus E5 | negative_precision | -0.027528 |
| E6 minus E5 | negative_recall | 0.118812 |
| E6 minus E5 | negative_f1 | 0.056107 |
| E6 minus E5 | accuracy | 0.076246 |
| E6 minus E5 | balanced_accuracy | -0.072173 |
| E6 minus E5 | negative_log_likelihood | -0.119255 |
| E6 minus E5 | brier_score | -0.049399 |
| E6 minus E5 | expected_calibration_error | -0.099443 |
| E6 minus E5 | roc_auc | -0.006601 |
| E6 minus E5 | average_precision | 0.076962 |
| E6 minus E5 | soft_brier_score | -0.044957 |
| E6 minus E5 | soft_cross_entropy | -0.106165 |
| E7 minus E6 | macro_f1 | -0.031773 |
| E7 minus E6 | positive_precision | -0.060098 |
| E7 minus E6 | positive_recall | 0.263158 |
| E7 minus E6 | positive_f1 | 0.005435 |
| E7 minus E6 | negative_precision | 0.026362 |
| E7 minus E6 | negative_recall | -0.138614 |
| E7 minus E6 | negative_f1 | -0.068981 |
| E7 minus E6 | accuracy | -0.093842 |
| E7 minus E6 | balanced_accuracy | 0.062272 |
| E7 minus E6 | negative_log_likelihood | 0.177592 |
| E7 minus E6 | brier_score | 0.065928 |
| E7 minus E6 | expected_calibration_error | 0.163335 |
| E7 minus E6 | roc_auc | 0.027097 |
| E7 minus E6 | average_precision | 0.101846 |
| E7 minus E6 | soft_brier_score | 0.059059 |
| E7 minus E6 | soft_cross_entropy | 0.158733 |
| E5 minus E1 | macro_f1 | 0.041534 |
| E5 minus E1 | positive_f1 | 0.161796 |
| E5 minus E1 | positive_recall | 0.526316 |
| E5 minus E1 | balanced_accuracy | 0.169098 |
| E5 minus E1 | accuracy | -0.108504 |
| E5 minus E1 | roc_auc | 0.065138 |
| E5 minus E1 | average_precision | 0.000886 |
| E5 minus E3 | macro_f1 | 0.019336 |
| E5 minus E3 | positive_f1 | 0.109614 |
| E5 minus E3 | positive_recall | 0.447368 |
| E5 minus E3 | balanced_accuracy | 0.141176 |
| E5 minus E3 | accuracy | -0.096774 |
| E5 minus E3 | roc_auc | 0.059232 |
| E5 minus E3 | average_precision | -0.067349 |
| E6 minus E2 | macro_f1 | -0.018277 |
| E6 minus E2 | positive_f1 | 0.005929 |
| E6 minus E2 | positive_recall | 0.184211 |
| E6 minus E2 | balanced_accuracy | 0.042600 |
| E6 minus E2 | accuracy | -0.067449 |
| E6 minus E2 | roc_auc | 0.024752 |
| E6 minus E2 | average_precision | -0.057612 |
| E6 minus E4 | macro_f1 | 0.006501 |
| E6 minus E4 | positive_f1 | 0.055840 |
| E6 minus E4 | positive_recall | 0.236842 |
| E6 minus E4 | balanced_accuracy | 0.065616 |
| E6 minus E4 | accuracy | -0.067449 |
| E6 minus E4 | roc_auc | 0.018586 |
| E6 minus E4 | average_precision | -0.036657 |

## Research questions

- RQ1 (context; E6−E5): mixed evidence on this locked test.
- RQ2 (soft-label training; E7−E6): mixed evidence on this locked test. Soft vote fractions are not ground-truth probabilities.
- RQ3 (calibration and abstention): mixed evidence on this locked test; interpret probability metrics and the frozen 0.60 operating point jointly with complete and class-specific coverage.

## Limitations

Only 38 positives occur in the test set. Findings are limited to AUTALIC and the frozen partition. Label 0 is not a definitive safe/non-ableist judgment; soft votes are descriptive annotator fractions. Exact-overlap grouping cannot detect every paraphrase. Calibration used 171 examples with 19 positives. Selective prediction may reject positive-labelled examples. No causal, universal-benefit, safety, or deployment-readiness claim is made.
