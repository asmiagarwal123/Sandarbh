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
