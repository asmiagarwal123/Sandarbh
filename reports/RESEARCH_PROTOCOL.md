# SANDARBH Research Protocol: Phase 2B

## Scope and approval record

This protocol freezes the dataset grouping, binary benchmark targets, and partition roles that precede model experiments. The researcher approved Phase 2A Rule B through the Phase 2B implementation prompt. Supporting text inspection was assistant-assisted; this is not a claim that the researcher personally completed all ten historical manual-review rows, which remain unchanged.

Rule B was approved because some short exact matches connect passages that Rule A separates, while accepting the conservative cost that generic greetings can also connect unrelated passages. Rule B is an exact-overlap proxy for leakage control. It does not reconstruct original Reddit posts and does not prevent semantic, paraphrastic, or other non-exact leakage.

No model training, learned preprocessing, tokenization, calibration, threshold selection, prediction, or evaluation occurs in this phase.

## Approved grouping policy

For eligible examples only, normalize each nonempty `preceding`, `target`, and `following` field with:

```python
" ".join(text.split()).casefold()
```

Connect different examples when any normalized field value is identical, regardless of field combination or word count. Take transitive connected components and retain unconnected examples as singleton groups. Empty strings never create edges. Each complete component must remain in one partition.

This transformation is for matching only and does not establish model-input preprocessing.

## Annotation preservation and binary targets

Every eligible example retains `A1_Score`, `A2_Score`, and `A3_Score`, the raw fractions for `-1`, `0`, and `1`, its three-way agreement category, and whether it contains a `-1` vote.

Define:

- `positive_vote_count`: number of annotations equal to `1`.
- `hard_label`: 1 if `positive_vote_count >= 2`, otherwise 0.
- `soft_positive`: `positive_vote_count / 3`.
- `soft_other`: `1 - soft_positive`.

Hard label 0 means “no majority ableist annotation under this benchmark mapping.” It is not a definitive “safe” or “not ableist” judgment. Both `-1` and `0` contribute to binary other, but their source distinction is retained because `-1` can indicate insufficient context or lack of relevance. The binary convention necessarily loses that distinction.

Rows with disagreement or `-1` votes are retained without relabeling, deletion, or weighting changes. An all-different `[-1, 0, 1]` annotation triple therefore yields hard label 0, soft positive `1/3`, and soft other `2/3`. Three-class training is outside Phase 2B.

Future hard- and soft-target experiments must use identical rows, partitions, input construction, and binary output space. Their later protocol may differ only in the declared training loss and target representation.

## Frozen partition roles

The outer allocation remains 70% train, 15% validation, and 15% test. Validation is divided into two disjoint group-constrained roles:

| Partition | Target | Outer split | Authorized use |
|---|---:|---|---|
| `train` | 0.700 | train | Fit models and every learned text-preprocessing object. |
| `dev_tune` | 0.075 | validation | Hyperparameters, early stopping, and checkpoint selection. |
| `dev_calibration` | 0.075 | validation | Fit a predeclared calibration procedure after model selection is frozen. |
| `test` | 0.150 | test | One final locked evaluation after methods and evaluation rules are frozen. |

`dev_calibration` must not influence model or checkpoint selection. Test predictions and performance must not influence any development choice. No additional small partitions may be created without an explicit later protocol amendment.

The dataset is small: each validation role may contain only roughly 19 hard-positive examples. Model-selection and calibration estimates may therefore be unstable. A later protocol must specify selective-prediction thresholds or their selection procedure before final test evaluation; no validated error guarantee is promised from these small partitions.

## Deterministic allocation

Configuration fixes base seed 42, 256 candidates, and seed stride 1,000,003. Candidate `i` uses a local `random.Random(42 + i * 1_000_003)`. Input groups are first canonicalized by group ID. Each candidate orders groups by descending size, then its deterministic random key, then group ID.

For each group, the greedy allocator tries each partition and selects the lexicographically smallest tuple of:

1. the provisional declared objective;
2. the candidate partition’s example fill ratio;
3. fixed partition order: train, dev_tune, dev_calibration, test.

The objective is the sum, across partitions and the example, hard-positive, and hard-other counts, of:

```text
(observed_count - target_count)^2 / max(target_count, 1)
```

Candidates are ranked by feasibility, total constraint violation, objective, and finally the canonical full assignment signature. Labels affect only these stratification counts; text meaning, review judgments, model predictions, and performance never affect allocation. Candidate count and seeds are fixed before execution, so there is no manual seed shopping.

A candidate is feasible only when every partition:

- is within 0.02 absolute of its target example fraction;
- has hard-positive rate within 0.03 absolute of the overall hard-positive rate;
- contains both hard classes.

Groups are never split to obtain exact percentages. If no candidate is feasible, the best candidate is recorded in `run_status.json`, the run is marked `NEEDS_REVIEW`, and no split is frozen.

## Publication, reuse, and validation

Successful outputs are staged and published only after the independent validator passes. The validator reads the immutable source and upstream artifacts, reconstructs targets from original annotations, reconstructs exact normalized overlaps from source text, and checks saved manifests rather than trusting summary success flags.

A successful frozen split cannot be silently overwritten. Identical code/config/input provenance triggers read-only validation and membership reuse. Different provenance stops with an explicit error; Phase 2B intentionally provides no force-overwrite option.

The validator’s precise overlap guarantee is: “No cross-partition nonempty exact normalized field overlap was found.” This must not be shortened to an unqualified “zero leakage” claim.

Generated artifacts contain IDs, targets, grouping, and partition roles but no raw text. Later modeling code must resolve IDs against the immutable source. Original source and Phase 1/2A artifacts remain unchanged.

## Decisions reserved for later phases

- Model families and exact input construction.
- Training-only learned preprocessing.
- Optimization, loss implementations, and hyperparameter spaces.
- Early-stopping details and checkpoint rules.
- Calibration family and fitting procedure.
- Selective-prediction thresholds or selection method.
- Metrics, uncertainty intervals, and final locked-test reporting rules.
