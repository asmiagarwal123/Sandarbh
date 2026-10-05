# SANDARBH Phase 3 Classical Baseline Protocol

Phase 3 implements four standard concatenated-text sparse baselines and one descriptive majority-class dummy benchmark. It uses the frozen Phase 2B hard labels and partitions without altering or regenerating them.

## Access boundary

The Phase 3 modeling loader permits only `train` and `dev_tune`. Vectorizers and classifiers fit exclusively on `train`. `dev_tune` is used for candidate scoring and selection. The interface rejects `dev_calibration` and `test`; neither is vectorized, predicted on, nor scored. Phase 3 does not refit a selected model on train plus development data.

Reading the frozen partition metadata and running its integrity validator does not authorize model-facing access to calibration or test text or labels.

## Input construction

Each source field is transformed with `" ".join(text.split())`: whitespace runs collapse to one ASCII space and outer whitespace disappears, while wording and punctuation remain. There is no stemming, lemmatization, stop-word removal, custom replacement, or truncation.

- Target-only input is the normalized `target` field.
- Context input joins the nonempty normalized `preceding`, `target`, and `following` fields, in that order, with a newline. The target occurs exactly once. Missing preceding or following context is allowed.

No labels, fractions, IDs, groups, or split names enter model input. These are ordinary concatenated-text baselines; TF-IDF does not explicitly encode sentence roles. Vectorizer lowercasing is distinct from the Phase 2 overlap-matching normalization and does not modify the immutable source.

## Experiments

| ID | Classifier | Input |
|---|---|---|
| E1 | Logistic Regression | target only |
| E2 | Logistic Regression | preceding + target + following |
| E3 | LinearSVC | target only |
| E4 | LinearSVC | preceding + target + following |

The dummy benchmark predicts the train-only majority class for every `dev_tune` example. It is descriptive and cannot be selected as a deployable project model.

## Fixed TF-IDF and candidate grid

Every experiment uses a scikit-learn Pipeline. TF-IDF is word-based, lowercases text, uses unigram/bigram features, `min_df=2`, `max_df=1.0`, `max_features=30000`, sublinear term frequency, L2 normalization, smoothed IDF, no stop-word list, float64 output, and the explicit standard token pattern `(?u)\b\w\w+\b`. Vocabulary, document frequencies, and IDF fit only on train.

Each experiment evaluates the fixed Cartesian grid:

- `C`: 0.1, 1.0, 10.0
- `class_weight`: null, `balanced`

This yields six candidates per experiment and 24 fits in the real run.

Logistic Regression uses the supported scikit-learn 1.9.1 expression of L2 regularization: `l1_ratio=0.0` with `solver="lbfgs"`, while leaving the deprecated `penalty` argument unset. It also uses `max_iter=5000`, `tol=0.0001`, and `random_state=42` (recorded although lbfgs does not use it).

LinearSVC uses L2 penalty, squared-hinge loss, `dual=True`, `max_iter=10000`, `tol=0.0001`, and `random_state=42`. Its decision-function output is a margin, never a probability.

## Selection and reporting

Each experiment is selected independently by exact, unrounded `dev_tune` macro-F1 over ordered labels `[0, 1]`; exact ties prefer smaller C and then null class weight before `balanced`. Predictions come from the estimator-native `predict` method. Decision thresholds are not tuned.

Metrics include macro-F1, positive-class precision/recall/F1, accuracy, balanced accuracy, `[0, 1]` confusion matrix, class support, average precision, and ROC-AUC where defined. Undefined precision/F1 uses `zero_division=0`; runs record when no positive is predicted. Logistic Regression ranking uses its positive-class probability, described as native and uncalibrated. LinearSVC ranking uses its margin.

Any scikit-learn convergence warning makes the run `NEEDS_REVIEW`; an affected candidate cannot silently become a valid winner, and iteration limits or solvers are not adjusted automatically.

Only selected-model development predictions and the four selected pipelines are saved. Serialized TF-IDF pipelines contain learned vocabulary terms and are kept local under ignored `models/`; they are not text-free artifacts. Reports and CSV/JSON outputs do not contain source sentences or vocabulary listings.

The 19 hard-positive `dev_tune` examples make differences potentially unstable. These are development estimates, not final generalization results. Phase 3 does not choose one overall project winner, evaluate test, calibrate probabilities, implement selective prediction, or begin transformer work.
