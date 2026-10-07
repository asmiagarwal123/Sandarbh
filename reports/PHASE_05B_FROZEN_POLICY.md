# SANDARBH Phase 5B Frozen Final-Evaluation Policy

## Status and boundary

Phase 5B converts the validated Phase 5A evidence and explicit research-controller decisions into an immutable machine-readable policy. A successful freeze has `policy_status = FROZEN` and `test_status = NOT_ACCESSED`. This phase performs no test-ID or label loading, test text inspection, tokenization, inference, prediction, metric calculation, retraining, or recalibration.

## Experiments and calibration

All three predesignated seed-42 experiments proceed to the one-time final evaluation: E5 target-only hard-label training at epoch 3, E6 context hard-label training at epoch 2, and E7 context soft-label training at epoch 2. No overall winner is selected. The controlled E6−E5 and E7−E6 comparisons answer the context and training-target questions respectively.

Scalar temperature scaling is frozen uniformly: E5 `1.0237168508133028`, E6 `0.6020567132971402`, and E7 `0.6154062772305571`. Phase 6 must retain uncalibrated probabilities for before/after probability-quality comparisons. The primary classification rule is calibrated argmax, equivalently positive probability at least 0.50, on the complete test set without abstention.

## Selective prediction

The shared secondary operating threshold is calibrated maximum-class confidence `0.60`. Confidence at least 0.60 is accepted; lower confidence abstains. This is a research operating point, not a safety guarantee. Calibrated thresholds 0.50, 0.60, 0.70, and 0.80 and the complete predeclared 0.50–0.99 grid are descriptive. Test results may never be used to replace the frozen 0.60 threshold. Class-specific coverage and adverse positive-label coverage tradeoffs must always be reported.

## Metrics, comparisons, and uncertainty

The primary metric is exact macro-F1 over labels `[0,1]`. The machine policy enumerates all required classification, probability-quality, soft-target descriptive, reliability, selective-risk, and Wilson-interval outputs. Empty bins and unavailable metrics remain unavailable, never zero. Hard label 0 is not a definitive safe judgment, and annotator vote fractions are not ground-truth probabilities.

Uncertainty uses 5,000 paired cluster-bootstrap replicates over frozen Rule B `group_id`, seed `20261006`, with 95% percentile intervals. Every sampled group contributes all member examples, and each replicate uses identical sampled groups and multiplicities across models. Single-class-unavailable values are recorded rather than resampled or changed to zero. Intervals are uncertainty summaries, not significance declarations; no p-values are generated.

## Immutability

The canonical policy object is serialized as UTF-8 JSON with sorted keys, compact separators, and no NaN, then SHA-256 hashed. Completed output cannot be silently overwritten. Any pre-test policy change requires an explicit new version; no policy change is permitted after test access. The only next research action is Phase 6’s one-time locked-test evaluation under this exact policy.
