# SANDARBH Phase 5A Calibration Protocol

## Scope and stopping rule

Phase 5A uses only the frozen 171-example `dev_calibration` partition and the predesignated E5, E6, and E7 seed-42 checkpoints. It generates probability-calibration and selective-prediction evidence for researcher review. It does not select a model, calibration version, confidence threshold, or deployment policy. It stops before test access and leaves `policy_status = PENDING_RESEARCHER_REVIEW`.

Hard label 0 means no majority ableist annotation under this benchmark mapping; it is not a definitive safe or non-ableist judgment. Annotator vote fractions are descriptive soft targets, not ground-truth probabilities.

## Locked inputs

- Frozen partition fingerprint: `61c9415766e4f7b342eb37df7b09adf2f89cd86f2818ee05645ae9409dbbba86`
- Model: `distilbert/distilroberta-base`
- Revision: `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`
- Checkpoints: `E5_seed42`, `E6_seed42`, and `E7_seed42`
- Device/precision: CPU/FP32
- Dynamic-padding inference batch size: 8
- Maximum sequence length: 256

The Phase 4B input constructor is reused. Whitespace is collapsed while wording, punctuation, and case are preserved. E5 uses the target-only encoding; E6 and E7 use byte-identical context encodings. Marker token assumptions (6/3/4), the 241-token fixed target budget, truncation, and context redistribution are revalidated before inference.

## Calibration

Each model receives one positive scalar temperature fitted to its 171 calibration logits and frozen hard labels by ordinary unweighted negative log-likelihood. A deterministic bounded golden-section search operates on log-temperature over `[0.05, 20.0]`, begins with identity as an explicit candidate, uses tolerance `1e-10`, and allows 500 iterations. Identity and temperature-scaled probabilities are the only versions. Scaling must preserve every argmax.

NLL, positive-probability Brier score, ten-bin equal-width ECE, ROC-AUC, average precision, native-argmax classification metrics, and descriptive soft-target agreement metrics are recorded. Empty reliability bins use unavailable values, never zero. Because the same small set fits and describes the calibrator, post-calibration differences are apparent fit-set diagnostics, not unbiased estimates.

## Selective prediction

Confidence is the maximum class probability. Thresholds `0.50` through `0.99` are evaluated for both probability versions. Outputs expose overall coverage, risk, Wilson 95% intervals, accepted confusion matrices, positive/negative-label coverage, and recall with abstentions treated as not automatically detected. Abstentions are never counted as correct.

Candidate evidence includes fixed thresholds 0.70/0.80/0.90 and lowest-risk grid points at minimum coverage 0.90/0.80/0.70, requiring accepted examples from both true classes. Ties prefer higher coverage then the lower threshold. Candidates are descriptive and unapproved.

## Integrity and publication

Upstream and checkpoint SHA-256 hashes are checked before and after inference. Outputs are staged and published atomically; a completed output directory is never overwritten. The independent validator reconstructs labels, metrics, reliability bins, risk/coverage, policy candidates, input encodings, and hashes. No raw text or token-ID sequence is published. No test tokenization, inference, prediction, inspection, thresholding, or metric calculation is permitted.
