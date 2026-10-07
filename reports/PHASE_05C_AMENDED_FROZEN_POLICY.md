# SANDARBH Phase 5C Amended Frozen Final-Evaluation Policy

## Amendment boundary

Phase 5C is an explicit pre-test successor to canonical Phase 5B policy `9c420cd4600bb50246977390aebe399aea143b836956a691e35c18d709db8f0c`. The original policy remains byte-identical and preserved. This amendment is permitted only because the original status is `NOT_ACCESSED`; it performs no test access, model loading, inference, retraining, recalibration, or Phase 6 work.

The successor schema is `5B-1.1.0`. Its sole substantive addition is the already-developed Phase 3 E1–E4 pipelines and deterministic majority-class dummy. Every transformer decision, temperature, threshold, metric, comparison, bootstrap setting, and interpretation restriction remains embedded unchanged from Phase 5B.

## Added frozen baselines

- E1: TF-IDF Logistic Regression, target input, hard labels, selected `C=10`, balanced class weights; native positive-class `predict_proba` output.
- E2: TF-IDF Logistic Regression, contextual input, hard labels, selected `C=0.1`, balanced class weights; native positive-class `predict_proba` output.
- E3: TF-IDF LinearSVC, target input, hard labels, selected `C=0.1`, balanced class weights; signed `decision_function` margin, never a probability.
- E4: TF-IDF LinearSVC, contextual input, hard labels, selected `C=0.1`, balanced class weights; signed `decision_function` margin, never a probability.
- DUMMY: always predicts hard label 0; classification metrics only, with probability and ranking outputs unavailable.

No vectorizer or classifier is refitted. Logistic Regression receives descriptive native-probability NLL and Brier metrics without calibration. LinearSVC receives ranking metrics from signed margins but no NLL, Brier, ECE, or probability interpretation. Unavailable values remain null.

## Comparisons and uncertainty

The successor retains E6−E5 and E7−E6 and adds E5−E1, E5−E3, E6−E2, and E6−E4 descriptive comparisons. No winner may be selected from test results.

The paired Rule B group bootstrap remains fixed at 5,000 replicates, seed `20261006`, and 95% percentile intervals. Identical sampled groups and multiplicities apply across E1–E7 and DUMMY. No p-values or resampling-until-valid are permitted.

## Immutability and next action

The successor policy is canonically serialized using the Phase 5B method and SHA-256 hashed. It does not overwrite `results/frozen_policy/`. Once validated, this successor becomes the only policy permitted for Phase 6’s one-time locked-test evaluation.
