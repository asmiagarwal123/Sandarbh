# SANDARBH Project Architecture

Generated from the project root on 2026-10-06. Every project file and folder name is listed. Git metadata, virtual-environment internals, and cache internals are intentionally collapsed; their root directory names remain visible.

```text
NLP/
|-- .git/
|   `-- [internal contents omitted]
|-- .venv/
|   `-- [internal contents omitted]
|-- .venv-transformers/
|   `-- [internal contents omitted]
|-- configs/
|   |-- audit.json
|   |-- baselines.json
|   |-- overlap_review.json
|   |-- partition.json
|   |-- transformer_preflight.json
|   `-- transformer_training.json
|-- models/
|   |-- baselines/
|   |   |-- E1_pipeline.joblib
|   |   |-- E2_pipeline.joblib
|   |   |-- E3_pipeline.joblib
|   |   `-- E4_pipeline.joblib
|   `-- transformers/
|       |-- E5_seed42/
|       |   |-- config.json
|       |   |-- model.safetensors
|       |   |-- tokenizer.json
|       |   `-- tokenizer_config.json
|       |-- E6_seed42/
|       |   |-- config.json
|       |   |-- model.safetensors
|       |   |-- tokenizer.json
|       |   `-- tokenizer_config.json
|       `-- E7_seed42/
|           |-- config.json
|           |-- model.safetensors
|           |-- tokenizer.json
|           `-- tokenizer_config.json
|-- Prompt_Packages/
|   |-- Prompt_01_Phase_03/
|   |   |-- configs/
|   |   |   `-- baselines.json
|   |   |-- reports/
|   |   |   |-- IMPLEMENTATION_STATUS.md
|   |   |   |-- PHASE_03_BASELINE_PROTOCOL.md
|   |   |   `-- phase_03_baselines.md
|   |   |-- results/
|   |   |   `-- baselines/
|   |   |       |-- candidate_metrics.csv
|   |   |       |-- dev_tune_predictions.csv
|   |   |       |-- dummy_metrics.json
|   |   |       |-- run_manifest.json
|   |   |       |-- run_status.json
|   |   |       `-- selected_models.json
|   |   |-- scripts/
|   |   |   |-- train_baselines.py
|   |   |   `-- validate_baseline_run.py
|   |   |-- tests/
|   |   |   `-- test_baselines.py
|   |   |-- PACKAGE_CONTENTS.md
|   |   |-- README.md
|   |   |-- requirements-baselines.txt
|   |   `-- SOURCE_PROMPT.txt
|   |-- Prompt_02_Phase_04A/
|   |   |-- benchmark.json
|   |   |-- environment.json
|   |   |-- PHASE_04_TRANSFORMER_PLAN.md
|   |   |-- phase_04a_transformer_preflight.md
|   |   |-- run_manifest.json
|   |   |-- run_status.json
|   |   |-- token_length_summary.json
|   |   |-- transformer_preflight.json
|   |   `-- truncation_examples.csv
|   `-- Prompt_03_Phase_04B/
|       |-- comparisons.json
|       |-- dev_tune_predictions.csv
|       |-- epoch_metrics.csv
|       |-- PHASE_04B_TRANSFORMER_PROTOCOL.md
|       |-- phase_04b_transformers.md
|       |-- run_manifest.json
|       |-- run_status.json
|       |-- seed_summary.json
|       |-- selected_checkpoint_metrics.csv
|       |-- tokenization_manifest.csv
|       |-- train_transformers.py
|       |-- transformer_training.json
|       `-- validate_transformer_run.py
|-- reports/
|   |-- IMPLEMENTATION_STATUS.md
|   |-- phase_01_dataset_audit.md
|   |-- phase_02a_overlap_review.md
|   |-- phase_02b_partitions.md
|   |-- PHASE_03_BASELINE_PROTOCOL.md
|   |-- phase_03_baselines.md
|   |-- PHASE_04_TRANSFORMER_PLAN.md
|   |-- phase_04a_transformer_preflight.md
|   |-- PHASE_04B_TRANSFORMER_PROTOCOL.md
|   |-- phase_04b_transformers.md
|   `-- RESEARCH_PROTOCOL.md
|-- results/
|   |-- audit/
|   |   |-- exclusions.csv
|   |   |-- overlap_links.csv
|   |   |-- row_manifest.csv
|   |   |-- run_status.json
|   |   `-- summary.json
|   |-- baselines/
|   |   |-- candidate_metrics.csv
|   |   |-- dev_tune_predictions.csv
|   |   |-- dummy_metrics.json
|   |   |-- run_manifest.json
|   |   |-- run_status.json
|   |   `-- selected_models.json
|   |-- overlap_review/
|   |   |-- group_membership.csv
|   |   |-- group_summary.csv
|   |   |-- manual_review.csv
|   |   |-- policy_comparison.json
|   |   `-- run_status.json
|   |-- partitions/
|   |   |-- run_status.json
|   |   |-- split_manifest.csv
|   |   |-- split_summary.json
|   |   `-- targets.csv
|   |-- transformer_preflight/
|   |   |-- benchmark.json
|   |   |-- environment.json
|   |   |-- run_manifest.json
|   |   |-- run_status.json
|   |   |-- token_length_summary.json
|   |   `-- truncation_examples.csv
|   `-- transformers/
|       |-- units/
|       |   |-- E5_seed42/
|       |   |   |-- epoch_metrics.csv
|       |   |   |-- predictions.csv
|       |   |   |-- selected_metrics.json
|       |   |   `-- unit.json
|       |   |-- E5_seed43/
|       |   |   |-- epoch_metrics.csv
|       |   |   |-- predictions.csv
|       |   |   |-- selected_metrics.json
|       |   |   `-- unit.json
|       |   |-- E5_seed44/
|       |   |   |-- epoch_metrics.csv
|       |   |   |-- predictions.csv
|       |   |   |-- selected_metrics.json
|       |   |   `-- unit.json
|       |   |-- E6_seed42/
|       |   |   |-- epoch_metrics.csv
|       |   |   |-- predictions.csv
|       |   |   |-- selected_metrics.json
|       |   |   `-- unit.json
|       |   |-- E6_seed43/
|       |   |   |-- epoch_metrics.csv
|       |   |   |-- predictions.csv
|       |   |   |-- selected_metrics.json
|       |   |   `-- unit.json
|       |   |-- E6_seed44/
|       |   |   |-- epoch_metrics.csv
|       |   |   |-- predictions.csv
|       |   |   |-- selected_metrics.json
|       |   |   `-- unit.json
|       |   |-- E7_seed42/
|       |   |   |-- epoch_metrics.csv
|       |   |   |-- predictions.csv
|       |   |   |-- selected_metrics.json
|       |   |   `-- unit.json
|       |   |-- E7_seed43/
|       |   |   |-- epoch_metrics.csv
|       |   |   |-- predictions.csv
|       |   |   |-- selected_metrics.json
|       |   |   `-- unit.json
|       |   `-- E7_seed44/
|       |       |-- epoch_metrics.csv
|       |       |-- predictions.csv
|       |       |-- selected_metrics.json
|       |       `-- unit.json
|       |-- comparisons.json
|       |-- dev_tune_predictions.csv
|       |-- epoch_metrics.csv
|       |-- run_manifest.json
|       |-- run_status.json
|       |-- seed_summary.json
|       |-- selected_checkpoint_metrics.csv
|       |-- tokenization_manifest.csv
|       |-- training_stderr.log
|       `-- training_stdout.log
|-- scripts/
|   |-- __pycache__/
|   |   `-- [internal contents omitted]
|   |-- audit_dataset.py
|   |-- prepare_partitions.py
|   |-- review_overlap_groups.py
|   |-- train_baselines.py
|   |-- train_transformers.py
|   |-- transformer_preflight.py
|   |-- validate_baseline_run.py
|   |-- validate_partitions.py
|   `-- validate_transformer_run.py
|-- tests/
|   |-- __pycache__/
|   |   `-- [internal contents omitted]
|   |-- test_baselines.py
|   |-- test_dataset_audit.py
|   |-- test_overlap_review.py
|   |-- test_partitions.py
|   |-- test_transformer_preflight.py
|   `-- test_transformer_training.py
|-- .gitignore
|-- AUTALIC.csv
|-- PROJECT_ARCHITECTURE.md
|-- README.md
|-- requirements.txt
|-- requirements-baselines.txt
`-- requirements-transformers.txt
```

Omitted internals are dependency, version-control, or generated cache implementation details rather than project source artifacts.
