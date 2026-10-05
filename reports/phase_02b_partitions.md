# Phase 02B Approved Targets and Partitions

**Status:** `FROZEN`  
**Generated (UTC):** 2026-10-05T04:01:07.959377Z  
**Approved grouping policy:** Rule B, approved by the user through the Phase 2B prompt after assistant-assisted text inspection. This does not claim that the user personally completed every historical manual-review case.

## Grouping policy

Rule B connects eligible examples sharing any nonempty exact normalized `preceding`, `target`, or `following` field and takes connected components including singletons. Matching uses `" ".join(text.split()).casefold()` only. It is a conservative exact-overlap proxy: generic greetings can overgroup unrelated passages, it does not reconstruct original posts, and it cannot remove every possible form of leakage.

## Binary benchmark targets

All three original annotations are retained. A vote is positive only when its value is `1`. `hard_label` is 1 when at least two votes are positive and 0 otherwise. Hard label 0 means **no majority ableist annotation under this benchmark mapping**; it is not a definitive safe or not-ableist judgment.

`soft_positive` is the fraction of annotations equal to 1 and `soft_other` is its complement. Both `-1` and `0` contribute to binary other, while their original columns and separate fractions remain available. A `-1` can reflect insufficient context or lack of relevance, so the binary convention loses a meaningful distinction. Disagreement and `-1` examples remain present without relabeling, dropping, or reweighting. Three-class training is outside this phase.

Hard and soft experiments must later use the same rows, partitions, input construction, and binary output space; only their declared loss/training targets may differ.

## Partition roles and observed counts

| Partition | Examples | Groups | Hard positive | Hard other | Has -1 |
|---|---:|---:|---:|---:|---:|
| train | 1595 | 998 | 180 | 1415 | 335 |
| dev_tune | 171 | 107 | 19 | 152 | 35 |
| dev_calibration | 171 | 107 | 19 | 152 | 42 |
| test | 341 | 213 | 38 | 303 | 71 |

- `train`: fit models and every learned text-preprocessing object.
- `dev_tune`: hyperparameters, early stopping, and checkpoint selection.
- `dev_calibration`: fit the predeclared calibration method only after model selection is frozen; never use it for checkpoint selection.
- `test`: one locked final evaluation after methods and evaluation rules are frozen; never use its predictions or performance for development decisions.

The dataset is small. In particular, the development and calibration roles may each contain only roughly 19 hard-positive examples; their model-selection and calibration estimates may be unstable. No extra tiny partitions were created.

## Allocation and validation

Groups were allocated by a deterministic 256-candidate greedy multi-start search using base seed 42. Candidate 188 (derived seed 188000606) minimized the predeclared count-balance objective among candidates, subject to fixed feasibility checks. Labels were used only for stratification and diagnostics.

Independent prepublication validation: `PASSED`. Guarantee: **No cross-partition nonempty exact normalized field overlap was found.** This is not an unqualified zero-leakage claim.

## Limitations and later decisions

- Rule B may overgroup generic short phrases and misses non-exact or semantic reuse.
- The binary target collapses the distinction between `-1` and `0` for training purposes, although the source distinctions remain recorded.
- A later protocol must predeclare selective-prediction thresholds or a selection method before final test evaluation; these small partitions do not support a promised error guarantee.
- Model families, learned preprocessing, hyperparameter spaces, calibration method, and evaluation rules remain for later locked protocols.

No model, tokenizer, preprocessing object, calibration model, threshold, prediction, or evaluation was produced in Phase 2B. No raw-text train/dev/test copies were saved.
