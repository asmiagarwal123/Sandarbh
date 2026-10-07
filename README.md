# SANDARBH

SANDARBH investigates anti-autistic ableist language detection using AUTALIC. Phase 1 provides a read-only, reproducible dataset audit before any split or modeling decision is made.

The audit validates CSV structure and labels, accounts for usable targets, summarizes the three annotations per example, measures text quality and length, and indexes exact target/context overlap. It never writes cleaned data and never changes `AUTALIC.csv`.

## Requirements and commands

Use Python 3.10 or newer. Phase 1 has no third-party dependencies. From the project root in Windows PowerShell:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/audit_dataset.py --config configs/audit.json
```

The audit resolves all configured paths relative to this project root, regardless of the shell's current directory. Exit codes are:

- `0`: completed without a blocking discrepancy (`PASS` or `PASS_WITH_WARNINGS`).
- `1`: completed, but a configured reference or source-integrity discrepancy requires review (`FAIL`).
- `2`: configuration, input, encoding, schema, label, or CSV parsing failure.

These statuses concern audit checks, not scientific validity or model quality. On every CLI attempt, `results/audit/run_status.json` identifies the latest run as `RUNNING`, `FAILED`, or `COMPLETED`; inspect it before trusting older outputs.

## Outputs

- `results/audit/summary.json`: machine-readable provenance and diagnostics.
- `results/audit/row_manifest.csv`: one text-free accounting record per source CSV record.
- `results/audit/exclusions.csv`: excluded IDs and reasons, with no raw text.
- `results/audit/overlap_links.csv`: hashed, text-free overlap evidence.
- `reports/phase_01_dataset_audit.md`: plain-language findings and limitations.

`source_row` is the one-based logical data-record index after the header. A CSV record may span multiple physical lines. Stable example IDs combine that index with a SHA-256 prefix of the original parsed field sequence. Rerun the same audit command to refresh outputs, then compare observed/reference entries in the summary and review warnings, failures, exclusions, and overlap implications in the report.

## Phase 2A: overlap-group policy review

After confirming that the Phase 1 run status is `COMPLETED`, prepare the two candidate overlap policies with:

```powershell
python scripts/review_overlap_groups.py --config configs/overlap_review.json
```

Rule A combines substantial exact overlap with an any-length repeated-target exception. Rule B uses every nonempty normalized exact overlap. Both retain singleton eligible examples, omit Phase 1 exclusions, and use matching-only whitespace normalization plus Unicode casefolding. Neither rule is automatically selected.

Phase 2A writes:

- `results/overlap_review/group_membership.csv`
- `results/overlap_review/group_summary.csv`
- `results/overlap_review/policy_comparison.json`
- `results/overlap_review/manual_review.csv`
- `results/overlap_review/run_status.json`
- `reports/phase_02a_overlap_review.md`

Inspect `run_status.json` before treating outputs as current. A successful run leaves policy status `PENDING_RESEARCHER_REVIEW`. Reviewer observations and notes in `manual_review.csv` are preserved by stable review case ID; if noted cases no longer map, the rerun stops and asks for a new output directory. No partition membership is produced.

## Phase 2B: approved targets and frozen partitions

The researcher subsequently approved Rule B. Phase 2B preserves the original three annotations, creates matched hard and soft binary benchmark targets, and allocates complete Rule B groups to train, tuning-development, calibration-development, or locked test roles.

From the project root, first run the complete tests, then prepare and independently validate the real outputs:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/prepare_partitions.py --config configs/partition.json
python scripts/validate_partitions.py --config configs/partition.json
```

Successful preparation creates:

- `results/partitions/targets.csv`
- `results/partitions/split_manifest.csv`
- `results/partitions/split_summary.json`
- `results/partitions/run_status.json`
- `reports/phase_02b_partitions.md`

`targets.csv` and `split_manifest.csv` contain no raw text. Hard label 0 means “no majority ableist annotation under this benchmark mapping,” not a definitive safe or not-ableist judgment. Both `-1` and `0` map to binary other, while their original distinctions remain recorded.

Inspect `run_status.json` for `COMPLETED`, then require the independent validator to exit successfully. Identical frozen provenance is validated and reused without changing membership. Different provenance stops rather than overwriting the split. There is intentionally no force-overwrite option.

The full protocol, locked-test policy, partition roles, deterministic objective, and remaining model-phase decisions are documented in `reports/RESEARCH_PROTOCOL.md`.

## Phase 3: classical TF-IDF baselines

Phase 3 implements four fixed development experiments:

- E1: TF-IDF + Logistic Regression, target only
- E2: TF-IDF + Logistic Regression, concatenated context
- E3: TF-IDF + LinearSVC, target only
- E4: TF-IDF + LinearSVC, concatenated context

It also reports a train-majority dummy benchmark on `dev_tune`. Vectorizers and classifiers fit only on `train`; the modeling loader rejects `dev_calibration` and `test`. Phase 3 does not perform calibration, selective prediction, transformer training, or final test evaluation.

The project-local Python 3.14 environment uses exact wheel-resolved dependencies in `requirements-baselines.txt`. Setup and execution from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-baselines.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe scripts/validate_partitions.py --config configs/partition.json
.\.venv\Scripts\python.exe scripts/train_baselines.py --config configs/baselines.json
.\.venv\Scripts\python.exe scripts/validate_baseline_run.py --config configs/baselines.json
```

The real training command writes text-free tabular results under `results/baselines/`, a report at `reports/phase_03_baselines.md`, and four selected pipelines under ignored `models/baselines/`. Serialized TF-IDF pipelines contain learned vocabulary terms and must remain local; they are not text-free artifacts.

Completed identical-provenance runs are validated and reused. Changed provenance requires a separately named run directory and is never silently overwritten. See `reports/PHASE_03_BASELINE_PROTOCOL.md` for the full fixed grid, metric definitions, selection rule, and locked partition boundary.

## Phase 4A: transformer preflight

Phase 4A uses the separate `.venv-transformers` environment to inspect the pinned `distilbert/distilroberta-base` tokenizer/model revision, measure actual token lengths on `train` and `dev_tune`, and run a bounded synthetic CPU optimizer benchmark. It does not fine-tune on AUTALIC or access `dev_calibration` or `test` through the tokenizer interface.

```powershell
.\.venv-transformers\Scripts\python.exe -m unittest tests.test_transformer_preflight -v
.\.venv-transformers\Scripts\python.exe scripts/transformer_preflight.py --config configs/transformer_preflight.json
.\.venv-transformers\Scripts\python.exe scripts/transformer_preflight.py --config configs/transformer_preflight.json --validate-only
```

The observed diagnostics and proposed target-preserving construction are documented in `reports/phase_04a_transformer_preflight.md` and `reports/PHASE_04_TRANSFORMER_PLAN.md`. This recommendation is not approval to begin E5–E7 training.

## Phase 4B: DistilRoBERTa development training

Phase 4B trains the fixed E5 target-only hard-label, E6 context hard-label, and E7 context soft-label configurations sequentially on CPU for seeds 42, 43, and 44. It uses the approved length-256 target-preserving construction and the immutable `distilbert/distilroberta-base` revision from Phase 4A. Seed 42 is predesignated for later stages; seeds 43 and 44 are development robustness runs only.

```powershell
.\.venv-transformers\Scripts\python.exe -m unittest tests.test_transformer_training -v
.\.venv-transformers\Scripts\python.exe scripts/train_transformers.py --config configs/transformer_training.json
.\.venv-transformers\Scripts\python.exe scripts/validate_transformer_run.py --config configs/transformer_training.json
```

The completed development results and protocol are in `reports/phase_04b_transformers.md` and `reports/PHASE_04B_TRANSFORMER_PROTOCOL.md`. Phase 4B uses only `train` and `dev_tune`; it does not access `dev_calibration` or `test`, perform calibration or threshold tuning, or make a final generalization claim.

## Phase 5A: calibration and selective-prediction review

Phase 5A runs the predesignated E5/E6/E7 seed-42 checkpoints once on the frozen `dev_calibration` partition, fits one scalar temperature per model, and creates text-free reliability, risk-coverage, and candidate-policy evidence. It stops with `PENDING_RESEARCHER_REVIEW`; it does not select a model or policy and does not access the test partition.

```powershell
.\.venv-transformers\Scripts\python.exe -m unittest tests.test_calibration_review -v
.\.venv-transformers\Scripts\python.exe scripts/run_calibration_review.py --config configs/calibration_review.json
.\.venv-transformers\Scripts\python.exe scripts/validate_calibration_review.py --config configs/calibration_review.json
```

See `reports/PHASE_05_CALIBRATION_PROTOCOL.md` for the locked protocol and `reports/phase_05a_calibration_review.md` for generated apparent fit-set diagnostics.

## Phase 5B: frozen final-evaluation policy

Phase 5B freezes all three seed-42 experiments, their scalar temperatures, calibrated-argmax primary classification, the shared 0.60 selective-confidence threshold, predeclared metrics/comparisons, and a 5,000-replicate paired Rule B group bootstrap. It performs no test access or model computation.

```powershell
.\.venv-transformers\Scripts\python.exe -m unittest tests.test_frozen_policy -v
.\.venv-transformers\Scripts\python.exe scripts/freeze_evaluation_policy.py --config configs/final_evaluation_policy.json
.\.venv-transformers\Scripts\python.exe scripts/validate_frozen_policy.py --config configs/final_evaluation_policy.json
```

The frozen policy and its canonical SHA-256 identifier are stored under `results/frozen_policy/`. Once frozen, the only permitted next research action is one-time locked-test evaluation under that exact policy.

## Phase 5C: pre-test baseline amendment

Before any test access, Phase 5C creates a versioned `5B-1.1.0` successor policy that retains Phase 5B byte-for-byte and adds the frozen Phase 3 E1–E4 pipelines plus the majority-class dummy to final evaluation. It does not overwrite the original policy or run any model.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_amended_policy -v
.\.venv\Scripts\python.exe scripts/amend_frozen_policy.py --config configs/final_evaluation_policy_v1_1.json
.\.venv\Scripts\python.exe scripts/validate_amended_policy.py --config configs/final_evaluation_policy_v1_1.json
```

After successful validation, the successor under `results/frozen_policy_v1_1/` is the only policy permitted for Phase 6.
# Phase 6 final evaluation

The frozen Phase 5C successor policy is evaluated exactly once by `scripts/evaluate_locked_test.py`. Use `--preflight-only` before the irreversible `--execute-once` mode. Final, text-free outputs are independently checked by `scripts/validate_final_test_run.py`; the protocol is in `reports/PHASE_06_FINAL_TEST_PROTOCOL.md`.

## Phase 7: final consolidation and local research UI

Phase 6 completed one locked evaluation of 341 test examples under successor policy `b5765e9431a4bf01e8abfeeec1762061b1c0a26eec4ea26c326abb7245ac7541`; its independent validator passed 42/42 checks. Phase 7 interprets those immutable outputs without recomputing predictions or changing the policy. The final report is `reports/FINAL_RESEARCH_REPORT.md`, publication tables are under ignored `results/final/tables/`, and text-free figures are under `reports/figures/`.

The three research questions concern context (classical and transformer target/context contrasts), soft-vote-fraction versus hard-label transformer training, and calibration/selective-prediction trade-offs. Evidence is mixed across metrics. The locked test has only 38 positive-labelled examples; class 0 means no majority positive annotation under the benchmark, not definitively safe or non-ableist.

The local Streamlit UI is a research demonstration of the frozen E5, E6, and E7 checkpoints. E6 is the neutral contextual default, not a test-selected winner. Setup and launch from PowerShell:

```powershell
.\.venv-transformers\Scripts\python.exe -m pip install -r requirements-ui.txt
.\.venv-transformers\Scripts\python.exe -m streamlit run app.py
```

The UI accepts optional preceding/following context and a required target sentence. It applies the exact 256-token input construction, frozen temperature, 0.50 class boundary, and 0.60 confidence threshold. It is not diagnostic, clinical, moderation-ready, or suitable for automated punishment. See `reports/UI_GUIDE.md`.

Final validation commands include:

```powershell
.\.venv-transformers\Scripts\python.exe -m unittest tests.test_ui tests.test_final_project -v
.\.venv-transformers\Scripts\python.exe scripts/validate_final_test_run.py --config configs/final_test_evaluation.json
.\.venv-transformers\Scripts\python.exe scripts/validate_final_project.py
```

`AUTALIC.csv`, `results/`, `models/`, checkpoints, virtual environments, caches, and Streamlit secrets are excluded from Git. Reproduction requires the original immutable source and locally retained frozen model artifacts.

## Version 1.1 forensic maintenance release

V1.1 (2026-10-07) preserves the complete scientific freeze and adds a forensic integrity validator, mathematical E7 loss audit, exact token-truncation audit, train-prevalence probability reference, dataset provenance review, seed summary, text-free post-hoc diagnostics, safer UI language, reproducibility locks/guidance, model card, research-question registry, Version 2 deferrals, and a hash-verified handoff package. No model was retrained or recalibrated, no partition/policy/checkpoint/prediction changed, and locked-test inference was not rerun.

The central errata are that E7 is a **cost-weighted soft-vote model**, not an unweighted vote-fraction learner; overlong targets retain head and tail around an ellipsis; and none of the available probability-producing frozen models beats the train-prevalence constant on locked-test Brier score. See `reports/V1_1_ERRATA_AND_VALIDATION.md` and `PROJECT_HANDOFF_V1_1.md`.

```powershell
& .\.venv-transformers\Scripts\python.exe scripts\validate_v1_1.py --run-synthetic-inference --finalize
```

The 0.60 value remains a research operating point, not a safety threshold. The dashboard is not a moderation system and class 0 does not prove harmlessness or acceptability.
