# SANDARBH Project Handoff for Codex

**Handoff version:** 1.0.0  
**Handoff date:** 2026-10-07 (Asia/Calcutta)  
**Repository root:** `C:\Users\asmi\Desktop\Projects\NLP`  
**Intended reader:** an AI agent with no prior project context

## 1. Executive status

SANDARBH is a leakage-controlled research project for binary detection of anti-autistic ableist language in the AUTALIC dataset. It compares target-only versus contextual inputs, hard versus soft annotation targets, scalar temperature calibration, and confidence-based selective prediction.

The research workflow is complete through Phase 7:

- The source dataset was audited without rewriting it.
- Exact-overlap groups and frozen train/development/calibration/test partitions were created.
- Four TF-IDF baselines (E1-E4) and three DistilRoBERTa experiments (E5-E7) were trained.
- E5-E7 were calibrated on the dedicated calibration partition.
- A versioned final-evaluation policy was frozen before test access.
- The 341-example locked test was evaluated exactly once.
- Final reports, figures, tables, a local Streamlit research UI, and a flattened handoff package were produced.

Current authoritative statuses:

| Component | Status | Evidence |
|---|---|---|
| Phase 1 audit | `COMPLETED`; audit result `PASS_WITH_WARNINGS` | `results/audit/run_status.json`, `results/audit/summary.json` |
| Phase 2A overlap review | `COMPLETED` | `results/overlap_review/run_status.json` |
| Phase 2B partitions | `COMPLETED`; partition summary `FROZEN`; internal validation `PASSED` | `results/partitions/run_status.json`, `results/partitions/split_summary.json` |
| Phase 3 baselines | `COMPLETED` | `results/baselines/run_status.json`, `results/baselines/run_manifest.json` |
| Phase 4A transformer preflight | `COMPLETED` | `results/transformer_preflight/run_status.json` and hash-verified copies in `Prompt_Packages/Prompt_02_Phase_04A/` |
| Phase 4B transformer training | `COMPLETED`, 9/9 units | `results/transformers/run_status.json`, `results/transformers/run_manifest.json` |
| Phase 5A calibration review | computation `COMPLETED`; historical review status `PENDING_RESEARCHER_REVIEW` | `results/calibration_review/run_status.json` |
| Phase 5B original policy | `FROZEN`, historical status `NOT_ACCESSED` | `results/frozen_policy/run_status.json` |
| Phase 5C successor policy | `FROZEN`, historical status `NOT_ACCESSED` | `results/frozen_policy_v1_1/run_status.json` |
| Phase 6 locked test | `COMPLETED`, `EVALUATED_ONCE` | `results/final_test/run_status.json` |
| Phase 6 validator | 42/42 passed | `results/final_test/validation_summary.json` |
| Phase 7 consolidation | `COMPLETED` | `results/final/final_summary.json` |
| Final-project validator | 39/39 passed | `results/final/final_validation_summary.json` |
| UI smoke receipts | passed | `results/final/ui_smoke_test.json`, `results/final/ui_server_smoke_test.json` |

The authoritative successor policy SHA-256 is:

```text
b5765e9431a4bf01e8abfeeec1762061b1c0a26eec4ea26c326abb7245ac7541
```

The test must not be evaluated again. Do not run Phase 6 with `--execute-once`.

## 2. Verification scope and caveats

This handoff was derived from the repository itself: top-level documentation, all configurations, source modules, scripts, tests, run statuses, manifests, machine-readable summaries, result tables, handoff inventories, dependency files, and model metadata were inspected. All non-environment project files were byte-read where Windows permissions allowed. Model and pipeline binaries were verified by SHA-256 rather than deserialized for this documentation task.

The following legacy files currently deny direct reads through their Windows ACLs:

```text
reports/phase_01_dataset_audit.md
reports/phase_02a_overlap_review.md
reports/phase_02b_partitions.md
results/transformer_preflight/benchmark.json
results/transformer_preflight/environment.json
results/transformer_preflight/run_manifest.json
results/transformer_preflight/token_length_summary.json
results/transformer_preflight/truncation_examples.csv
```

For the five Phase 4A result files, readable copies under `Prompt_Packages/Prompt_02_Phase_04A/` match the reference SHA-256 values frozen in `configs/transformer_training.json`. Their substantive values are therefore documented below. The exact current contents of the three inaccessible early Markdown reports are **UNKNOWN / NEEDS VERIFICATION**; all Phase 1/2 facts in this handoff are instead supported by readable configurations and JSON/CSV artifacts.

Virtual-environment internals, Git object internals, and Python cache internals were not reviewed file by file. Installed dependency versions were queried from both local environments, and `pip check` succeeded in each.

## 3. Purpose, questions, and interpretation boundary

The project asks three research questions:

1. Does conversational context change performance relative to target-only input?
2. How does soft-vote-fraction training compare with hard-label training when contextual input construction is identical?
3. How do frozen scalar temperature calibration and confidence-based abstention affect probability quality, coverage, and observed selective risk?

This is research software, not a safety product. It is not validated for diagnosis, clinical use, automated moderation, punishment, or other consequential decisions. Class 0 means only “no majority ableist annotation under this benchmark mapping”; it does not prove that language is safe or non-ableist. Soft vote fractions are annotator fractions, not ground-truth probabilities.

## 4. Architecture and end-to-end workflow

```text
AUTALIC.csv
  -> Phase 1: schema, label, quality, and overlap audit
  -> Phase 2A: candidate exact-overlap grouping policies
  -> Phase 2B: approved Rule B groups, binary targets, frozen partitions
       |-> Phase 3: E1-E4 TF-IDF baselines on train/dev_tune
       |-> Phase 4A: DistilRoBERTa tokenizer and CPU feasibility preflight
       `-> Phase 4B: E5-E7 training on train/dev_tune
             -> Phase 5A: seed-42 inference and temperature fitting on dev_calibration
             -> Phase 5B: original frozen transformer evaluation policy
             -> Phase 5C: pre-test successor adding E1-E4 and DUMMY
             -> Phase 6: one-time locked-test evaluation
             -> Phase 7: reports, tables, figures, validation, UI, handoff
```

Architectural properties:

- Paths are resolved relative to the repository root.
- Generated modeling artifacts are local and Git-ignored.
- Stable example IDs combine the one-based logical CSV record index with a SHA-256 prefix of the original parsed fields.
- Text is joined back to IDs only inside partition-authorized loaders.
- Published manifests and prediction tables omit raw source text.
- Upstream files, configurations, scripts, and models are hash-checked before later phases use them.
- Completed runs use provenance fingerprints, atomic publication, and overwrite refusal or exact-provenance reuse.
- Development tuning, calibration, and locked testing use separate frozen partitions.
- The final test policy was frozen before the one-time test access receipt was created.

## 5. Dataset, annotations, preprocessing, and partitions

### 5.1 Source dataset

`AUTALIC.csv` is UTF-8 with BOM and has this exact six-column schema:

```text
preceding,target,following,A1_Score,A2_Score,A3_Score
```

Verified source properties:

| Property | Value |
|---|---:|
| Source SHA-256 | `66ccaa43f9f3d8694c4826d8b999a4e562041637cad3c8c396590fbe78d3d408` |
| File size | 834,207 bytes |
| Raw records | 2,400 |
| Eligible records with a nonempty target | 2,278 |
| Excluded records | 122 |
| Exclusion reason | all three text fields empty |
| All three text fields present | 1,876 |
| Target and following present, preceding absent | 402 |
| Exact duplicate eligible triplets | 0 |

Phase 1 did not rewrite the source. The audit summary records identical pre/post hashes.

### 5.2 Raw annotation values and benchmark targets

Each example has three annotations: `A1_Score`, `A2_Score`, and `A3_Score`. Allowed raw values are `-1`, `0`, and `1`.

The frozen binary mapping is:

- A positive vote is an annotation equal to `1`.
- Hard label 1: at least two of three annotations equal `1`.
- Hard label 0: otherwise; both raw `-1` and raw `0` count as binary “other.”
- `soft_positive = positive_vote_count / 3`.
- `soft_other = 1 - soft_positive`.
- Original annotations and their `-1` versus `0` distinction remain in `results/partitions/targets.csv`.

Eligible-set counts:

| Item | Count |
|---|---:|
| Hard label 1 | 256 |
| Hard label 0 | 2,022 |
| Unanimous annotations | 1,310 |
| Two-versus-one | 884 |
| All different | 84 |
| Eligible examples containing at least one `-1` | 483 |

### 5.3 Normalization and overlap control

Audit/grouping normalization is matching-only:

```python
" ".join(text.split()).casefold()
```

It collapses Unicode whitespace, strips outer whitespace, and applies Unicode casefolding. It is not a source rewrite and is distinct from model input preprocessing.

Rule A groups substantial exact overlaps (minimum five words) plus the repeated-target exception. Rule B groups every nonempty normalized exact overlap. Rule B was approved for partitioning.

Verified Rule B statistics:

| Item | Value |
|---|---:|
| Eligible examples | 2,278 |
| Total Rule B groups | 1,425 |
| Singleton groups | 988 |
| Non-singleton groups | 437 |
| Largest group | 22 examples |

Rule B keeps connected components intact across partitions. The partition validator recorded no cross-partition nonempty exact normalized field overlap. This controls detected exact overlap only; it does not detect all paraphrases or semantic near-duplicates and may overgroup generic short strings.

### 5.4 Frozen partitions

The configured proportions are 70% train, 7.5% development tuning, 7.5% development calibration, and 15% test. Whole Rule B groups were allocated using 256 deterministic candidate assignments and a squared normalized count-deviation objective. Candidate 188, derived seed `188000606`, was selected with no constraint violation.

| Partition | Examples | Rule B groups | Hard 0 | Hard 1 |
|---|---:|---:|---:|---:|
| `train` | 1,595 | 998 | 1,415 | 180 |
| `dev_tune` | 171 | 107 | 152 | 19 |
| `dev_calibration` | 171 | 107 | 152 | 19 |
| `test` | 341 | 213 | 303 | 38 |
| **Total** | **2,278** | **1,425** | **2,022** | **256** |

Frozen partition fingerprint:

```text
61c9415766e4f7b342eb37df7b09adf2f89cd86f2818ee05645ae9409dbbba86
```

Relevant artifact hashes:

```text
results/partitions/targets.csv
e72ca8a447f91428d83e51217cedec90950bb2d2c4dfb64ad1ae338f40e7fd76

results/partitions/split_manifest.csv
fef7b3c036a5841bb13aadf0271f0da18b81f03fa4ca44b86300e3d08463aa76
```

## 6. Models

### 6.1 Classical baselines

All four baselines are scikit-learn pipelines fitted only on `train` and selected only on `dev_tune`.

Common TF-IDF settings:

- Word analyzer, lowercase enabled.
- Unigrams and bigrams.
- `min_df=2`, `max_df=1.0`, `max_features=30000`.
- Sublinear term frequency, smoothed IDF, L2 normalization.
- No stop-word list.
- Float64 output.
- Token pattern `(?u)\b\w\w+\b`.

Input normalization is `" ".join(text.split())`; context mode joins nonempty preceding, target, and following fields with newline separators.

| ID | Model | Input | Frozen selected candidate | Vocabulary |
|---|---|---|---|---:|
| E1 | TF-IDF + Logistic Regression | target | `C=10`, `class_weight=balanced` | 6,201 |
| E2 | TF-IDF + Logistic Regression | context | `C=0.1`, `class_weight=balanced` | 21,258 |
| E3 | TF-IDF + LinearSVC | target | `C=0.1`, `class_weight=balanced` | 6,201 |
| E4 | TF-IDF + LinearSVC | context | `C=0.1`, `class_weight=balanced` | 21,258 |
| DUMMY | Always predicts class 0 | none | train-majority definition | n/a |

The candidate grid was `C in {0.1, 1.0, 10.0}` crossed with `class_weight in {null, balanced}`, producing 24 fits. Selection used exact unrounded `dev_tune` macro-F1, then smaller C, then null class weight before balanced. Selected models were not refitted on combined train/development data.

Frozen baseline pipeline hashes:

```text
E1  1e87824ca409500932348c0f0d9992f60e14b1fac36b598a40318cef9df67bd3
E2  4a97c435c77235f019c5328fe8c7fb1025807418402d85c71f35ad76cbff7c2d
E3  13a0d8544e1d61fc58dc53222137e1d7b05d381cc93bcf172dac67fcef3521a1
E4  076895be3321c70a379970414e1a790bddad8638485978b971d794fcf6bb1c29
```

E1/E2 expose native uncalibrated positive-class probabilities. E3/E4 expose signed SVC margins, not probabilities. DUMMY has neither probability nor ranking output.

### 6.2 Transformer experiments

E5-E7 use `distilbert/distilroberta-base` at immutable revision:

```text
fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b
```

The model is a six-layer RoBERTa sequence classifier with hidden size 768, 12 attention heads, 50,265-token vocabulary, two labels, and FP32 weights.

| ID | Input | Training target/loss | Primary seed | Selected epoch |
|---|---|---|---:|---:|
| E5 | target only | class-weighted hard cross-entropy | 42 | 3 |
| E6 | context | class-weighted hard cross-entropy | 42 | 2 |
| E7 | identical contextual encoding to E6 | class-weighted soft cross-entropy | 42 | 2 |

Seeds 42, 43, and 44 were trained for each experiment. Seed 42 was predesignated for calibration and final evaluation; seeds 43/44 were robustness runs and could not replace it based on development performance. Only the three selected seed-42 checkpoints are retained.

Frozen checkpoint weight hashes:

```text
E5  96c100bfce0c547f385636ecc9f15b6e2772cf7ca3d995442543ec9aecb0ba9e
E6  ff5d477ab875e5d487f1fee9c329c419bdfdaecf50e15da2f5edda3f250c4b49
E7  ea85aef0135587f1390de5e8ed5f6b6ab7f342136bf263184c2ce212ba4e29a2
```

### 6.3 Exact transformer input construction

Fields are normalized with `" ".join(text.split())`; case, punctuation, and wording are preserved. Content and exact markers are tokenized separately without automatic special tokens:

```text
Preceding context:\n
Target:\n
Following context:\n
```

The final maximum is 256 tokens including one BOS and one EOS token. The marker token lengths are 6, 3, and 4, leaving a fixed target-content cap of 241 after reserving all markers and outer special tokens.

- E5: `BOS + target marker + retained target + EOS`.
- E6/E7: `BOS + preceding marker + preceding content + target marker + retained target + following marker + following content + EOS`.
- A target of at most 241 tokens is kept completely.
- An overlong target keeps its beginning, tokenized literal ` …`, and ending; the beginning receives the extra token when capacity is odd.
- Remaining context capacity is split equally; preceding keeps the tail nearest the target, following keeps the head nearest the target, and unused capacity transfers deterministically to the other side.
- E6 and E7 use byte-identical input IDs and attention masks.
- Dynamic padding pads only to the longest item in a microbatch and never beyond 256.

The UI calls the same `construct_encodings` implementation from `scripts/train_transformers.py`; it does not maintain a separate approximation.

## 7. Training, calibration, policy freezing, and evaluation

### 7.1 Transformer optimization

Verified Phase 4B settings:

- CPU FP32.
- Three epochs for every experiment/seed.
- Microbatch 2, gradient accumulation 8, effective batch 16.
- AdamW learning rate `2e-5`, weight decay `0.01`.
- Gradient clipping norm `1.0`.
- Linear decay after 10% warmup.
- 798 microbatches and 100 optimizer updates per epoch; 300 updates total.
- Final partial accumulation groups are scaled and stepped, not discarded.
- No early stopping, parameter freezing, gradient checkpointing, threshold tuning, ensembling, or hyperparameter search.
- Train hard-label counts `[1415, 180]`; class weights are `N / (2 * class_count)` and shared across E5-E7.
- Each experiment/seed selects the highest exact unrounded `dev_tune` macro-F1; ties use the earliest epoch.

### 7.2 Calibration

Phase 5A ran only the three predesignated seed-42 checkpoints on all 171 `dev_calibration` examples, including 19 hard positives. It fitted one bounded scalar temperature per model by golden-section search over log-temperature, optimizing unweighted hard-label NLL over `[0.05, 20.0]` with tolerance `1e-10` and limit 500 iterations.

| Model | Frozen temperature | Selected epoch |
|---|---:|---:|
| E5 | 1.0237168508133028 | 3 |
| E6 | 0.6020567132971402 | 2 |
| E7 | 0.6154062772305571 | 2 |

Temperature scaling is:

```text
calibrated probabilities = softmax(original logits / temperature)
```

It preserves argmax. Phase 5A calibration improvements are apparent fit-set diagnostics because the same 171 examples fit and describe the calibrator.

### 7.3 Frozen decision policy

The original Phase 5B policy hash is:

```text
9c420cd4600bb50246977390aebe399aea143b836956a691e35c18d709db8f0c
```

Phase 5C created version `5B-1.1.0`, adding the already-frozen E1-E4 pipelines and DUMMY before test access. It did not overwrite the original. The successor hash `b576...7541` is the only authoritative Phase 6 policy.

Frozen transformer decision logic:

- Primary classification: calibrated argmax at full coverage, equivalent to class 1 when calibrated `p1 >= 0.50`.
- Confidence: `max(p0, p1)`.
- Frozen selective operating point: accept when confidence `>= 0.60`; otherwise abstain.
- Descriptive grid: thresholds 0.50 through 0.99 in steps of 0.01.
- Primary metric: macro-F1 over ordered labels `[0, 1]`.
- Uncertainty: paired complete-Rule-B-group bootstrap, 5,000 replicates, seed `20261006`, 95% percentile intervals.
- Bootstrap uses identical sampled groups and multiplicities across systems.
- No p-values, no resampling until valid, and unavailable values remain unavailable rather than zero.
- No model winner is selected.

### 7.4 One-time test evaluation

Phase 6 evaluated exactly 341 test examples, with 38 hard positives and 213 Rule B groups. The permanent receipt was created immediately before access. Current status is `EVALUATED_ONCE`.

Receipt file SHA-256:

```text
9569a832f9ebf85f6e497aabecaeba029f5b9c37f24ce69085622631ad6f4f4e
```

Never delete, replace, or regenerate `results/final_test/test_access_receipt.json`. Never rerun `scripts/evaluate_locked_test.py --execute-once`.

## 8. Current locked-test results

### 8.1 Classification and ranking

| System | Macro-F1 | Positive precision | Positive recall | Positive F1 | Accuracy | Balanced accuracy | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DUMMY | 0.4705 | 0.0000 | 0.0000 | 0.0000 | 0.8886 | 0.5000 | unavailable | unavailable |
| E1 | 0.5769 | 0.3043 | 0.1842 | 0.2295 | 0.8622 | 0.5657 | 0.7005 | 0.2698 |
| E2 | 0.6539 | 0.5882 | 0.2632 | 0.3636 | 0.8974 | 0.6200 | 0.7343 | 0.4053 |
| E3 | 0.5991 | 0.3030 | 0.2632 | 0.2817 | 0.8504 | 0.5936 | 0.7064 | 0.3381 |
| E4 | 0.6291 | 0.6154 | 0.2105 | 0.3137 | 0.8974 | 0.5970 | 0.7405 | 0.3843 |
| E5 | 0.6184 | 0.2700 | 0.7105 | 0.3913 | 0.7537 | 0.7348 | 0.7657 | 0.2707 |
| E6 | 0.6356 | 0.3148 | 0.4474 | 0.3696 | 0.8299 | 0.6626 | 0.7591 | 0.3477 |
| E7 | 0.6039 | 0.2547 | 0.7105 | 0.3750 | 0.7361 | 0.7249 | 0.7862 | 0.4495 |

### 8.2 Transformer probability quality

| Model | State | NLL | Brier | ECE | Soft Brier |
|---|---|---:|---:|---:|---:|
| E5 | uncalibrated | 0.5059 | 0.1694 | 0.2338 | 0.1177 |
| E5 | calibrated | 0.5056 | 0.1691 | 0.2357 | 0.1173 |
| E6 | uncalibrated | 0.4371 | 0.1337 | 0.2076 | 0.0816 |
| E6 | calibrated | 0.3864 | 0.1197 | 0.1362 | 0.0723 |
| E7 | uncalibrated | 0.5808 | 0.1952 | 0.3320 | 0.1382 |
| E7 | calibrated | 0.5640 | 0.1856 | 0.2996 | 0.1314 |

Calibration improved E6/E7 NLL, Brier, and ECE. E5 had very small NLL/Brier improvements and slightly worse ECE.

### 8.3 Frozen selective operating point

| Model | Accepted | Coverage | Selective risk | Positive-label coverage | Negative-label coverage |
|---|---:|---:|---:|---:|---:|
| E5 | 289 | 0.8475 | 0.2249 | 0.7895 | 0.8548 |
| E6 | 305 | 0.8944 | 0.1311 | 0.8684 | 0.8977 |
| E7 | 279 | 0.8182 | 0.2151 | 0.7895 | 0.8218 |

Lower observed risk occurs with reduced coverage and is not a safety guarantee.

### 8.4 Research-question conclusions

- **RQ1, context:** mixed. Classical context increased locked-test macro-F1 for E2 versus E1 and E4 versus E3. E6 versus E5 changed macro-F1 by `+0.0172` but positive recall by `-0.2632`; the recall bootstrap interval was `[-0.5000, -0.0833]`.
- **RQ2, soft targets:** mixed. E7 versus E6 changed macro-F1 by `-0.0318`, positive F1 by `+0.0054`, and positive recall by `+0.2632`; the recall interval was `[0.1304, 0.4286]`. E7 improved ranking metrics while worsening several probability-quality measures relative to E6.
- **RQ3, calibration and abstention:** mixed. Calibration helped E6/E7 most clearly. The 0.60 policy lowers observed error among accepted cases but removes examples from both classes, including positive-labelled cases.

These are descriptive results for AUTALIC and this frozen split. They do not establish universal superiority, statistical significance, causal effects, deployment readiness, or safety.

## 9. Streamlit dashboard

### 9.1 Entry point and tabs

The entry point is `app.py`. It has four tabs:

1. Prediction
2. Model information
3. Research results
4. Limitations and ethics

The aggregate-results tab renders three validated figures and does not expose source examples or individual saved predictions.

### 9.2 Inputs

- Frozen experiment selector: E5, E6, or E7.
- Optional preceding context.
- Required target sentence.
- Optional following context.

E6 is the default only as a neutral contextual demonstration; it is not a test-selected winner. E5 uses target encoding and ignores context for inference. E6/E7 use contextual encoding.

Empty or whitespace-only target input raises `ValueError("Target sentence is required.")` and is displayed as a UI error.

The Clear button uses a pre-render callback and resets only `preceding`, `target`, and `following`. It does not reset model selection or alter any scientific setting.

### 9.3 Inference path

`src/sandarbh_ui/inference.py`:

- Allows exactly E5/E6/E7.
- Verifies the selected `model.safetensors` SHA-256 before loading.
- Loads tokenizer and model from local checkpoint directories with `local_files_only=True` and `trust_remote_code=False`.
- Moves the model to CPU, converts it to FP32, and calls `model.eval()`.
- Uses `torch.inference_mode()`.
- Applies exact Phase 4B target-preserving construction through `src/sandarbh_ui/input_builder.py` and `scripts/train_transformers.py`.
- Applies the frozen model-specific temperature.
- Predicts class 1 when `p1 >= 0.50`.
- Accepts when maximum calibrated probability is at least 0.60; otherwise returns `ABSTAIN`.
- Caches at most three loaded model bundles with `lru_cache` plus Streamlit resource caching.
- Does not log or write entered text.

### 9.4 Outputs

The Prediction tab shows:

- `ACCEPTED` or `ABSTAIN`.
- Calibrated class-1 probability.
- Calibrated class-0 probability.
- Confidence.
- Plain-language benchmark-class explanation and research-use warning.
- Experiment description, whether context was used, temperature, CPU inference time, and encoded length out of 256.

## 10. Repository map and important files

| Path | Purpose |
|---|---|
| `AUTALIC.csv` | Immutable source dataset; local and Git-ignored |
| `README.md` | Phase-by-phase operational overview |
| `PROJECT_ARCHITECTURE.md` | Older architecture snapshot; see known issues because it is stale |
| `app.py` | Streamlit entry point |
| `configs/*.json` | Frozen phase configurations and provenance references |
| `scripts/audit_dataset.py` | Phase 1 audit |
| `scripts/review_overlap_groups.py` | Phase 2A grouping comparison |
| `scripts/prepare_partitions.py` | Phase 2B target and split construction |
| `scripts/train_baselines.py` | E1-E4 development training |
| `scripts/transformer_preflight.py` | Phase 4A diagnostics and synthetic benchmark |
| `scripts/train_transformers.py` | E5-E7 encoding, training, metrics, and checkpoint selection |
| `scripts/run_calibration_review.py` | Phase 5A inference, temperature fitting, reliability, and risk/coverage |
| `scripts/freeze_evaluation_policy.py` | Phase 5B original policy freeze |
| `scripts/amend_frozen_policy.py` | Phase 5C successor policy creation |
| `scripts/evaluate_locked_test.py` | Phase 6 evaluator; irreversible execution mode must never be rerun |
| `scripts/validate_*.py` | Independent validators for saved artifacts |
| `scripts/build_final_project.py` | Derives Phase 7 summaries, reports, tables, and figures from saved Phase 6 outputs |
| `scripts/build_final_handoff.py` | Builds and hash-indexes the flattened final handoff |
| `src/sandarbh_ui/` | UI encoding, frozen inference, formatting, and Clear callback |
| `tests/` | Phase-specific unit and integrity tests |
| `models/baselines/` | Four local serialized TF-IDF pipelines |
| `models/transformers/` | Three local frozen seed-42 checkpoints and tokenizers |
| `results/audit/` | Text-free audit manifests and summary |
| `results/overlap_review/` | Rule A/B membership and review artifacts |
| `results/partitions/` | Frozen targets and split manifest |
| `results/baselines/` | Development baseline metrics, predictions, manifests |
| `results/transformers/` | Nine-unit development results and text-free tokenization metadata |
| `results/calibration_review/` | Phase 5A probabilities, temperatures, reliability and risk/coverage |
| `results/frozen_policy*/` | Original and successor frozen policies |
| `results/final_test/` | Immutable Phase 6 receipt, predictions, metrics, intervals, and validation |
| `results/final/` | Phase 7 final summary, publication tables, and validation receipts |
| `reports/FINAL_RESEARCH_REPORT.md` | Main research narrative |
| `reports/figures/` | Six PNG and six SVG final figures |
| `reports/UI_GUIDE.md` | UI usage and interpretation guide |
| `Prompt_Packages/` | Phase-specific delivery bundles; final flattened package is `Prompt_07_Phase_06_07_Final` |

## 11. Environments and dependencies

### 11.1 Base/audit environment

Phase 1 requires Python 3.10+ and only the standard library. The currently installed project environments both report Python 3.14.0.

### 11.2 Baseline environment

Existing executable:

```text
.\.venv\Scripts\python.exe
```

Exact pinned packages from `requirements-baselines.txt`:

```text
cloudpickle==3.1.2
joblib==1.6.0
narwhals==2.26.0
numpy==2.5.3
scikit-learn==1.9.1
scipy==1.18.1
threadpoolctl==3.7.0
```

Documented setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-baselines.txt
```

### 11.3 Transformer/UI environment

Existing executable:

```text
.\.venv-transformers\Scripts\python.exe
```

Verified principal versions:

```text
Python 3.14.0
torch 2.14.1+cpu
transformers 5.18.0
tokenizers 0.23.2
safetensors 0.8.0
numpy 2.5.3
psutil 7.2.2
streamlit 1.65.0
matplotlib 3.11.2
CUDA available: false
```

The repository does not document the original command that created `.venv-transformers`: **UNKNOWN / NEEDS VERIFICATION**.

The exact Phase 4A configuration records these installation commands:

```powershell
.\.venv-transformers\Scripts\python.exe -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
.\.venv-transformers\Scripts\python.exe -m pip install --only-binary=:all: --index-url https://pypi.org/simple transformers==5.18.0 safetensors psutil
```

`requirements-transformers.txt` pins the complete transformer environment and includes the CPU PyTorch index. UI dependencies are bounded separately:

```text
streamlit>=1.50,<2
matplotlib>=3.10,<4
```

Install UI dependencies into the transformer environment with:

```powershell
.\.venv-transformers\Scripts\python.exe -m pip install -r requirements-ui.txt
```

Both environments currently pass `pip check`.

## 12. Exact operational commands

Run commands from the repository root in Windows PowerShell.

### 12.1 Safe current inspection and validation

```powershell
# Unit tests
.\.venv-transformers\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v

# Existing dependency checks
.\.venv\Scripts\python.exe -m pip check
.\.venv-transformers\Scripts\python.exe -m pip check

# Read-only locked-test validation
.\.venv-transformers\Scripts\python.exe scripts\validate_final_test_run.py --config configs\final_test_evaluation.json

# UI-focused tests
.\.venv-transformers\Scripts\python.exe tests\test_ui.py -v
.\.venv-transformers\Scripts\python.exe tests\test_final_project.py -v

# Repository hygiene
git diff --check
git status --short
```

`scripts/validate_final_project.py` is labeled `read_only_except_summary` and rewrites `results/final/final_validation_summary.json`. Run it only when updating that generated validation receipt is intended:

```powershell
.\.venv-transformers\Scripts\python.exe scripts\validate_final_project.py
```

### 12.2 Launch the UI

```powershell
.\.venv-transformers\Scripts\python.exe -m streamlit run app.py
```

Streamlit normally serves `http://localhost:8501`. Stop it with `Ctrl+C`.

### 12.3 Historical phase commands

These commands document how completed phases were produced. Do not rerun training, calibration, partition creation, policy freezing, final-project generation, or handoff generation unless a new explicitly authorized workflow requires it.

```powershell
python scripts\audit_dataset.py --config configs\audit.json
python scripts\review_overlap_groups.py --config configs\overlap_review.json
python scripts\prepare_partitions.py --config configs\partition.json
python scripts\validate_partitions.py --config configs\partition.json

.\.venv\Scripts\python.exe scripts\train_baselines.py --config configs\baselines.json
.\.venv\Scripts\python.exe scripts\validate_baseline_run.py --config configs\baselines.json

.\.venv-transformers\Scripts\python.exe scripts\transformer_preflight.py --config configs\transformer_preflight.json
.\.venv-transformers\Scripts\python.exe scripts\transformer_preflight.py --config configs\transformer_preflight.json --validate-only
.\.venv-transformers\Scripts\python.exe scripts\train_transformers.py --config configs\transformer_training.json
.\.venv-transformers\Scripts\python.exe scripts\validate_transformer_run.py --config configs\transformer_training.json

.\.venv-transformers\Scripts\python.exe scripts\run_calibration_review.py --config configs\calibration_review.json
.\.venv-transformers\Scripts\python.exe scripts\validate_calibration_review.py --config configs\calibration_review.json
```

The pre-access Phase 5B and 5C validators assert that test artifacts do not exist. They are expected to fail after the legitimate Phase 6 evaluation and are not valid current health checks.

### 12.4 Prohibited Phase 6 command

The evaluator historically supported:

```text
scripts/evaluate_locked_test.py --execute-once
```

Do not run it. The permanent receipt and `EVALUATED_ONCE` status prove that the allowed test access has already occurred. The only appropriate Phase 6 command now is the read-only validator shown above.

## 13. Completed work and recorded validation

Completed work includes:

- Read-only dataset audit and text-free manifests.
- Rule A/Rule B overlap review and Rule B approval.
- Frozen group-preserving four-way split.
- 24 fixed baseline candidate fits and E1-E4 selection.
- Transformer tokenizer diagnostics and CPU feasibility benchmark.
- Nine complete transformer training units: E5/E6/E7 × seeds 42/43/44.
- Retention of only selected seed-42 E5/E6/E7 checkpoints.
- 513 Phase 5A calibration prediction rows, six calibration metric rows, 60 reliability rows, and 300 risk/coverage rows.
- Original Phase 5B policy and versioned Phase 5C successor.
- One-time Phase 6 evaluation of E1-E7 plus DUMMY.
- 5,000-replicate paired Rule B cluster bootstrap.
- Final report, eight final tables, six PNG and six SVG figures.
- Local Streamlit UI with frozen inference and aggregate-only result display.
- Final flattened 42-file handoff package with hash inventory.
- Clear-input callback correction and focused regression test.

The recorded final execution summary reports:

- Phase 6 validator: 42/42 passed.
- Final-project validator: 39/39 passed.
- Focused Phase 6/7/UI tests: 69 passed.
- Full unittest discovery at final consolidation: 226 passed with five environment-expected skips.
- Both environment dependency checks passed.
- Real UI inference and controlled Streamlit startup smoke tests passed.

These counts are recorded historical receipts, not tests rerun while creating this handoff.

## 14. Design decisions that must be preserved

1. **No raw source rewriting.** Normalization is applied at matching or model-input time only.
2. **Rule B group isolation.** Every nonempty exact normalized overlap forms connected groups that cannot cross partitions.
3. **Separate development roles.** `dev_tune` selects candidates/epochs; `dev_calibration` fits temperatures; `test` is one-time only.
4. **Stable benchmark mapping.** Raw `-1` and `0` both map to binary other while remaining preserved in targets.
5. **Predesignated transformer seed.** Seed 42 is authoritative; robustness seeds cannot replace it after viewing development results.
6. **Target-preserving tokenization.** The 256-token construction protects target content and nearby context deterministically.
7. **E6/E7 input identity.** They differ in training target/loss, not contextual encoding.
8. **Full-coverage primary evaluation.** Selective prediction is secondary and cannot replace primary classification results.
9. **Shared 0.60 confidence threshold.** It may not be changed based on test behavior.
10. **No post-test winner selection.** All systems and trade-offs are reported; no universal winner is declared.
11. **Group bootstrap without p-values.** Intervals describe uncertainty and are not significance claims.
12. **Local-only sensitive artifacts.** Source data, learned vocabularies, checkpoints, results, environments, and secrets remain Git-ignored.
13. **No user-text persistence in the UI.** User input remains in memory and is not logged or written by application code.

## 15. Immutable and protected components

Do not modify, replace, refit, or regenerate these components in the completed project:

- `AUTALIC.csv` and its SHA-256.
- `results/audit/`, `results/overlap_review/`, and the approved Rule B membership.
- `results/partitions/targets.csv`, `split_manifest.csv`, and partition fingerprint.
- All four baseline pipelines and their hashes.
- The three retained seed-42 transformer checkpoints, tokenizers, selected epochs, and hashes.
- Frozen calibration temperatures.
- Original Phase 5B policy and successor Phase 5C policy.
- The authoritative successor policy hash.
- The 0.50 classification boundary and 0.60 selective threshold.
- Metric definitions, comparisons, bootstrap seed/settings, and interpretation restrictions.
- `results/final_test/test_access_receipt.json`.
- All Phase 6 predictions, metrics, intervals, manifests, statuses, and validation receipts.
- Phase 7 reports, tables, figures, and handoff unless explicitly producing a new separately versioned derivative from existing text-free results.

Any future scientific change requires a new version, new external data, and a separately preregistered evaluation. It must not reuse the locked test for tuning or another evaluation.

## 16. Limitations and known issues

### Scientific limitations

- The dataset is class-imbalanced: only 256/2,278 eligible examples are hard-positive.
- `dev_tune` and `dev_calibration` each contain only 19 positives; the test contains only 38.
- Calibration diagnostics on `dev_calibration` are fit-set diagnostics, not unbiased validation estimates.
- Exact-overlap grouping cannot detect all paraphrases or semantic duplicates and can merge generic short strings.
- Results apply only to AUTALIC and this frozen split.
- Annotator votes are not ground-truth probabilities.
- Confidence abstention can reject positive-labelled cases and cannot guarantee safety.
- Errors in either direction can be harmful.
- Bootstrap intervals depend on the observed Rule B groups and are not p-values.

### Operational and repository issues

- Eight legacy files have Windows ACL read-deny behavior, listed in Section 2. Phase 4A copies were hash-verified; the exact three early report texts remain **UNKNOWN / NEEDS VERIFICATION**.
- `PROJECT_ARCHITECTURE.md` was generated before later Phase 5-7 additions and is now incomplete/stale. Use this handoff plus the live filesystem inventory instead.
- `results/calibration_review_invalid_ece_schema/` and `results/calibration_review_invalid_validator/` are non-canonical extra directories. Their intended lifecycle is not documented: **UNKNOWN / NEEDS VERIFICATION**. Do not use them as scientific evidence; the canonical directory is `results/calibration_review/`.
- The Git working tree contains extensive modified and untracked work from completed phases. It is not a clean committed snapshot.
- Model checkpoints are roughly 328 MB each and are omitted from Git and the flattened final handoff. Reproduction requires the retained local artifacts or a separately controlled reconstruction.
- Baseline joblib files contain learned vocabulary terms and are not text-free.
- The UI is CPU-only and the first request for each model includes checkpoint-loading latency.
- The original transformer virtual-environment creation command is not recorded: **UNKNOWN / NEEDS VERIFICATION**.
- A fully clean external reproduction has not been demonstrated from the flattened final handoff alone because source data, environments, baseline pipelines, and transformer checkpoints are intentionally excluded.

## 17. Future work

Future scientific work should be a new, separately versioned project stage and should not reopen the locked test. Repository-supported recommendations are:

- Collect or obtain new external evaluation data.
- Increase positive-class sample size.
- Add broader semantic/paraphrase overlap detection while controlling false grouping.
- Conduct independent annotation and adjudication studies.
- Evaluate calibration on larger independent data.
- Pre-register new model, threshold, and metric decisions before accessing new evaluation data.
- Validate fairness, robustness, domain transfer, and human-review workflows before considering any consequential use.
- Improve environment bootstrap documentation and resolve legacy ACL problems without altering protected artifact bytes.
- Create a new architecture inventory if desired; do not silently rewrite the historical `PROJECT_ARCHITECTURE.md` snapshot.

## 18. Guidance for the next AI agent

Before doing work:

1. Read this file, `README.md`, `reports/FINAL_RESEARCH_REPORT.md`, and the relevant phase protocol.
2. Inspect `git status --short`; preserve all existing user changes.
3. Identify whether the request is documentation/UI maintenance or a new scientific phase.
4. For any scientific work, verify immutable hashes and confirm that it does not access or retune against the locked test.
5. Prefer existing validators and text-free summaries over inspecting individual test records.
6. Never infer that class 0 means safe, that soft votes are truth probabilities, or that the highest test score defines a winner.

Safe default for maintenance work: change only the specifically requested non-scientific files, run focused tests, run the read-only Phase 6 validator if relevant, and document any validator that rewrites a receipt before invoking it.

## 19. Primary evidence used for this handoff

The main evidence sources were:

- `configs/audit.json`, `overlap_review.json`, `partition.json`, `baselines.json`, `transformer_preflight.json`, `transformer_training.json`, `calibration_review.json`, `final_evaluation_policy*.json`, and `final_test_evaluation.json`.
- `results/audit/summary.json`, overlap/partition summaries, baseline selection artifacts, transformer manifests/metrics, calibration parameters/metrics, frozen-policy envelopes, Phase 6 metrics/receipt/validation, and Phase 7 summary/validation.
- `scripts/` and `tests/`, including the exact UI inference and target-preserving input path.
- `README.md`, phase protocols, `reports/FINAL_RESEARCH_REPORT.md`, and `reports/UI_GUIDE.md`.
- Installed environment version queries and SHA-256 checks of all seven retained model artifacts.

Where readable narrative reports and machine-readable artifacts differed in authority, frozen configurations, hashes, manifests, and validators were treated as authoritative.
