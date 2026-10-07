# SANDARBH V1.1 errata and forensic validation

Version: 1.1.0  
Date: 2026-10-07  
Status: maintenance addendum; historical scientific artifacts are unchanged.

## Scientific freeze

V1.1 did not retrain or refit E1-E7, change partitions, labels, selected epochs, primary seeds, temperatures, the 0.50 decision boundary, the frozen 0.60 research operating point, comparisons, bootstrap settings, checkpoints, predictions, or the test-access receipt. The validator hashes 130 immutable files before and after maintenance. It does not read AUTALIC as CSV, locked-test sentences, or rerun locked-test inference.

## Corrections and clarifications

E7 is a **cost-weighted soft-vote model**. It forms `[soft_other, soft_positive]`; multiplies both components by class weights; sums class loss per example; and averages examples. With training counts 1,415/180, weights are 0.5636042402826855 and 4.430555555555555. The optimum is

`p1*(q) = w1 q / (w0(1-q) + w1 q)`.

Thus q = 0, 1/3, 2/3, 1 maps to 0, 0.7971830986, 0.9401993355, 1. E7 is not ordinary vote-fraction cross-entropy, does not isolate soft labels, and E7-versus-E6 cannot establish a causal soft-label effect.

Overlong targets use a head-plus-tokenized-ellipsis-plus-tail construction, not left-prefix truncation. The 256-token contextual layout reserves BOS/EOS plus 6 preceding-marker, 3 target-marker, and 4 following-marker tokens, leaving a 241-token maximum target content budget. Remaining context capacity is split evenly, with an odd token assigned to preceding; unused capacity is redistributed. Preceding keeps its tail and following keeps its head. Saved hashes show identical E6/E7 contextual encodings.

The Brier score is binary positive-class mean squared error. The train prevalence is 0.1128526646 and locked-test prevalence is descriptively 0.1114369501. The train-prevalence reference has NLL 0.3495191683 and Brier 0.0990207605. All available probability-producing models (E1, E2, E5, E6, E7) have negative Brier skill; E3/E4 expose decision margins, not probabilities. Some models retain ranking ability. Accordingly: **The model shows ranking/discrimination ability but does not demonstrate probability skill over the train-prevalence reference under this scoring rule.**

The local AUTALIC file has 2,400 rows, of which 122 are all-empty and excluded; the 2,278 eligible rows yield 256 positives under the documented binary mapping. The paper reports 2,400 targets, 2,014 preceding contexts, 2,400 following contexts, 242 positives, and average Fleiss' kappa about 0.25. Its 242 positive plus 2,160 negative statement sums to 2,402. Local availability is 2,278 target, 1,876 preceding, and 2,278 following fields. Official byte identity and retrieval metadata remain unknown and need author verification. A1/A2/A3 are annotation positions across three 800-item segments with different trios, not three identifiable annotators spanning the dataset.

## Interpretation addendum

- E2 has the highest observed macro-F1 (0.653907), but is not a proven winner; numerical differences are underpowered.
- Accuracy is misleading under approximately 11% positive test prevalence.
- Context changed the operating behavior, especially false-positive/recall trade-offs, but V1 does not demonstrate correct contextual reasoning.
- Exact-overlap grouping does not rule out paraphrase, thread, author, or semantic leakage.
- Cluster bootstrap intervals condition on fixed trained models and exclude training-seed variability. No p-values were added.
- V1 is descriptive research, not deployment evidence.

## Threshold and lifecycle clarification

The 0.60 confidence rule was added explicitly before the one-time test execution in Phase 5C. Phase 5A examined fixed 0.70/0.80/0.90 candidates and constrained lowest-risk candidates plus risk-coverage tables on development calibration data. Phase 5B froze the original policy; Phase 5C amended it pre-test with 0.60 as a fixed research operating point. It is not a safety threshold and was not reselected in V1.1.

Phase 5B/5C validators enforce pre-access lifecycle states and may now reject a legitimately completed post-test repository. `calibration_review_invalid_ece_schema` and `calibration_review_invalid_validator` are retained negative-test fixtures/evidence, not active calibration runs. Phase 5A is complete as a calibration review; it did not approve a deployment policy. Phase 6 used the `.venv-transformers` interpreter with CPU PyTorch/Transformers plus the saved baseline dependencies; exact historical executable path is recorded in the Phase 6 manifest where available.

## Exploratory V1.1 findings

All are **POST-HOC / EXPLORATORY / NOT USED FOR MODEL SELECTION**. Exact confusion matrices, agreement/context/group-size strata, entropy associations, model-disagreement tables, risk at 0.60, and the existing frozen intervals are in `results/v1_1_validation/posthoc_summary.json`. Only the paired difference intervals already present for macro-F1, positive F1, and positive recall are reported as paired intervals; requested additional paired intervals are explicitly unavailable rather than retroactively extending the frozen bootstrap.

## Remaining limitations

The corpus is US/Reddit-specific, small, imbalanced, and context-sensitive. Temperature scaling did not establish probability skill over the prevalence reference. The selective threshold is a research operating point. Synthetic smoke cases validate software paths only—not accuracy, fairness, safety, or scientific validity.

## Validation command map

Full discovery in the baseline environment: 240 passed, 6 expected transformer/UI skips. Focused transformer/UI/V1.1 suite: 30 passed. Active validators passed 20 partition, 17 baseline, 14 preflight, 17 transformer, 30 calibration, and 42 final-test checks. V1.1 verified 152 immutable files pre/post, ran 33 authored synthetic model paths, verified all 13 required package copies, and completed. The historical Phase 7 final-project package hash check is expected to fail after V1.1 UI/source maintenance and is not a current gate; all 38 non-package-identity checks passed.
