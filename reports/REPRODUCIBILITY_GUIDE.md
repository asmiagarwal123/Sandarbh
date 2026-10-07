# SANDARBH V1.1 reproducibility guide (Windows)

Version 1.1.0 — 2026-10-07

## Prerequisites and placement

Use 64-bit Python 3.14.0 and PowerShell from the repository root. Obtain AUTALIC from its authors under their restrictions and place it at `AUTALIC.csv`; verify the expected local hash before doing anything. Place frozen local transformer directories at `models/transformers/E5_seed42`, `E6_seed42`, and `E7_seed42`; each must contain config/tokenizer files and `model.safetensors`. Baseline pipelines belong under `models/baselines`. Do not download models during validation.

## Environments

```powershell
py -3.14 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-baselines.txt

py -3.14 -m venv .venv-transformers
& .\.venv-transformers\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-transformers\Scripts\python.exe -m pip install --prefer-binary --progress-bar off -r requirements.txt
```

The captured 2026-10-07 baseline freeze is cloudpickle 3.1.2, joblib 1.6.0, narwhals 2.26.0, numpy 2.5.3, scikit-learn 1.9.1, scipy 1.18.1, threadpoolctl 3.7.0. Core transformer/UI packages are numpy 2.5.3, pandas 3.0.6, torch 2.14.1+cpu, transformers 5.18.0, tokenizers 0.23.2, safetensors 0.8.0, Streamlit 1.65.0, matplotlib 3.11.2, psutil 7.2.2. Full captured outputs are in `requirements-baselines-lock.txt` and `requirements-transformers-lock.txt`.

## Active commands

```powershell
# All active tests
& .\.venv-transformers\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v

# Active validators (never add --execute-once)
& .\.venv\Scripts\python.exe scripts\validate_partitions.py --config configs\partition.json
& .\.venv\Scripts\python.exe scripts\validate_baseline_run.py --config configs\baselines.json
& .\.venv-transformers\Scripts\python.exe scripts\validate_transformer_run.py --config configs\transformer_training.json
& .\.venv-transformers\Scripts\python.exe scripts\validate_final_test_run.py --config configs\final_test_evaluation.json
& .\.venv-transformers\Scripts\python.exe scripts\validate_v1_1.py --run-synthetic-inference --finalize

# CPU-only local UI
$env:CUDA_VISIBLE_DEVICES=""
& .\.venv-transformers\Scripts\python.exe -m streamlit run app.py
```

Historical lifecycle validators `validate_frozen_policy.py`, `validate_amended_policy.py`, and some Phase 5 pre-access checks are evidence of their original states; after legitimate one-time test access they are not current-state release gates. `validate_final_project.py` validates byte identity of the historical Phase 7 package, including UI/source files intentionally superseded by V1.1, so its handoff-hash check is expected to fail after V1.1 maintenance; use the passing final-test and V1.1 validators as current gates.

## Determinism and clean-room checklist

- Confirm source/config/artifact hashes before execution.
- Confirm partitions, labels, seed 42 primary checkpoints, epochs, temperatures, 0.50 boundary, 0.60 threshold, bootstrap seed 20261006, 5,000 replicates, and group clustering.
- Work offline with local checkpoint/tokenizer files; CPU float32 inference is the release path.
- Run `pip check` in both environments and record `pip freeze`.
- Run active tests/validators and `git diff --check`.
- Scan new reports/JSON/CSV/package files for source-text leakage.
- Compare `pre_change_integrity.json` and `post_change_integrity.json`.
- Confirm packaged copies against `PACKAGE_CONTENTS.md`.
- Do not rerun test inference, regenerate the receipt, retrain, recalibrate, or alter frozen artifacts.
- Do not commit AUTALIC, checkpoints, environments, caches, logs, prediction-level test rows, or secrets.
