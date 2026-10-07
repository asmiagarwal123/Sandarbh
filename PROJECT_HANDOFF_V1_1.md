# SANDARBH V1.1 project handoff

Handoff version: 1.1.0  
Date: 2026-10-07  
Audience: an AI or human maintainer with zero prior context  
Release state: final Version 1 forensic maintenance; consult `results/v1_1_validation/run_status.json` for the executed status.

## 1. Purpose and boundaries

SANDARBH is a descriptive NLP research project comparing target-only and context-aware classifiers for majority ableism annotation under the AUTALIC benchmark's binary mapping. It studies classical baselines (E1-E4), DistilRoBERTa target/context models (E5-E7), temperature scaling, and confidence-based selective prediction. It is not a diagnostic, clinical, safety, moderation, punishment, or deployment system.

Version 1 has completed one locked-test evaluation. The scientific state is immutable: do not retrain/refit, alter data or partitions, change labels, epochs, seeds, checkpoints, temperatures, the 0.50 classification boundary, the 0.60 selective threshold, metrics, comparisons, bootstrap settings, predictions, receipt, or policy; do not rerun `evaluate_locked_test.py --execute-once`; do not inspect locked-test sentences. V1.1 adds only text-free audits, tests, wording, documentation, reproducibility metadata, and synthetic authored smoke tests. No Version 2 work was performed.

## 2. Current lifecycle

Phases 1-4B audited data, grouped overlap, froze partitions, trained baselines, performed transformer preflight, and trained nine development transformer runs. Phase 5A used only dev-calibration to fit one temperature per seed-42 transformer and describe candidate operating points. Phase 5B froze the transformer policy. Phase 5C, before test access, created successor policy 5B-1.1.0 adding baselines/dummy. Phase 6 evaluated the locked test exactly once; the active validator passes 42/42. Phase 7 consolidated aggregate results and the local UI. V1.1 forensically checks that state and records errata.

Phase 5A's computation is `COMPLETED` while its original policy disposition was `PENDING_RESEARCHER_REVIEW`; Phase 5B/5C subsequently froze the approved research policy. Historical pre-access validators can fail after legitimate test access and are not present-state gates. The `calibration_review_invalid_ece_schema` and `calibration_review_invalid_validator` directories are retained negative-test evidence and must not be mistaken for active results.

## 3. Repository architecture

- `AUTALIC.csv`: private/restricted source; hash-only access in V1.1 validation.
- `configs/`: phase configurations and `v1_1_validation.json`.
- `scripts/`: phase runners/validators. `validate_v1_1.py` is the current V1.1 builder/validator.
- `models/baselines/`: frozen E1-E4 joblib pipelines.
- `models/transformers/`: frozen seed-42 E5/E6/E7 config, tokenizer, and weights.
- `results/`: ignored local scientific artifacts. `final_test/` is immutable; `v1_1_validation/` contains derivative text-free audits.
- `reports/`: immutable phase reports plus additive V1.1 errata/model/data/reproducibility documents.
- `src/sandarbh_ui/`: local inference, input construction, and safe result formatting.
- `app.py`: Streamlit research demonstration.
- `tests/`: phase and active V1.1 unit tests.
- `Prompt_Packages/`: ignored private handoff copies; the V1.1 package is `Prompt_06_V1_1_Final_Validation`.

Data flow: private CSV → text-free audit manifest → exact-overlap groups → group-aware partitions/targets → baseline and transformer development → dev-calibration temperature scaling/policy freeze → one-time locked-test predictions → aggregate Phase 7 and V1.1 text-free analyses. Source text is not copied into final results or packages.

## 4. Dataset, labels, preprocessing, and splits

The local file is 834,207 bytes, SHA-256 `66ccaa43f9f3d8694c4826d8b999a4e562041637cad3c8c396590fbe78d3d408`, with 2,400 rows and six expected fields: preceding, target, following, A1, A2, A3 (see the audit for exact header spelling). Download source/date and official byte equality are **UNKNOWN / NEEDS VERIFICATION**.

Saved audit facts: 122 records have all three text fields empty and are excluded; 2,278 are eligible. Local nonempty counts are target 2,278, preceding 1,876, following 2,278. Annotation value 1 maps positive; 0 and -1 map other; hard class 1 requires at least two positive votes. Soft positive is positive-vote count / 3 and soft other is 1-soft-positive. Eligible hard positives: 256. A1/A2/A3 are role columns across three 800-item annotation segments with different trios, not stable annotator identities.

The group-aware split contains train 1,595, dev-tune 171, dev-calibration 171, and locked test 341. Test has 38 positives and 303 other (11.1437% positive), with 213 groups. Train has 180 positives and 1,415 other (11.2853% positive). Exact-overlap groups prevent exact normalized duplicates crossing partitions but do not rule out paraphrase, thread, author, or semantic leakage.

Classical preprocessing is fixed within saved pipelines/configs. Transformer normalization collapses whitespace while preserving case. The model revision is `distilbert/distilroberta-base` commit `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`. Maximum encoded length is 256.

## 5. Experiments and training

- E1/E2: target-only classical pipelines with native positive probabilities in final outputs.
- E3/E4: contextual classical pipelines exposing decision margins, not probabilities.
- E5: target-only DistilRoBERTa, hard labels, seed 42 epoch 3.
- E6: preceding + target + following, hard labels, seed 42 epoch 2.
- E7: identical contextual input to E6, cost-weighted soft-vote targets, seed 42 epoch 2.

All transformer development ran seeds 42, 43, and 44; only seed-42 selected checkpoints are retained. Seed 42 was designated before test access. Development robustness is descriptive because n=3.

E7 code constructs `[soft_other, soft_positive]` and computes `-(target * class_weight * log_probability).sum(classes).mean(batch)`. Training counts 1,415/180 imply weights 0.5636042402826855/4.430555555555555. For vote fraction q, its optimum is `w1*q / (w0*(1-q)+w1*q)`: 0, 0.797183, 0.940199, 1 for q 0, 1/3, 2/3, 1. It is not ordinary soft cross-entropy and does not directly learn the raw vote fraction. E7-E6 does not isolate a causal soft-label effect.

## 6. Exact token construction

Contextual input is BOS + 6-token preceding marker + retained preceding + 3-token target marker + retained target + 4-token following marker + retained following + EOS. Two outer plus 13 marker tokens leave at most 241 target content tokens. Overlong target: head + tokenized literal ` …` + tail, with an odd available content token assigned to the head. Remaining context capacity splits evenly (odd to preceding); if one side is short/missing, unused capacity goes to the other. Preceding retains its tail nearest target; following retains its head nearest target. Saved train/dev manifests show E6/E7 hashes identical. Seven ordinary pre-construction combined sequences exceeded 256; the actual marker-aware construction filled length 256 for 10 records, partially truncating two targets and never completely removing a target. Detailed locked-test truncation counts were not saved and remain unavailable; V1.1 did not load test text.

## 7. Calibration, evaluation, and results

Each E5-E7 temperature is fitted on dev-calibration by bounded scalar hard-label NLL minimization: E5 1.0237168508133028, E6 0.6020567132971402, E7 0.6154062772305571. Temperature scaling preserves argmax. Primary class boundary is 0.50. The 0.60 confidence threshold is a pre-test research operating point, not a safety guarantee.

Locked-test macro-F1: E1 0.576912, E2 0.653907, E3 0.599110, E4 0.629129, E5 0.618446, E6 0.635630, E7 0.603857. E2 is numerically highest but is not a proven winner. Transformer positive precision/recall: E5 0.270/0.711, E6 0.315/0.447, E7 0.255/0.711. Exact confusion matrices are in `posthoc_summary.json`.

Probability metrics use binary positive-class Brier, hard-label NLL, 10 equal-width p1-bin ECE, rank AUC with average tie ranks, and grouped-tie AP. Train-prevalence null p=0.1128526646 has NLL 0.3495191683 and Brier 0.0990207605. Brier skill is negative for E1 (-0.0629), E2 (-1.2573), E5 (-0.7079), E6 (-0.2090), and E7 (-0.8748). E3/E4 have no probability metrics. Models can rank examples while failing to show probability skill over this reference.

The paired cluster bootstrap uses group ID, seed 20261006, and 5,000 replicates. Paired difference intervals exist only for macro-F1, positive F1, and positive recall. Model intervals cover the metrics recorded in `bootstrap_intervals.csv`. It conditions on fixed trained models and omits training-seed variability; no p-values exist.

At threshold 0.60, coverage/risk are E5 0.8475/0.2249, E6 0.8944/0.1311, E7 0.8182/0.2151. These are exploratory summaries, not guarantees.

## 8. Streamlit behavior

The UI accepts required target plus optional context, permits explicit E5/E6/E7 selection, defaults neutrally to E6, loads local checkpoints lazily on CPU, validates weight hashes, applies the exact token builder and frozen temperature, predicts at 0.50, then assigns an above/below-0.60 review status. It now separately displays predicted class, model-estimated probability, confidence, and review status. It never calls above-threshold output “accepted,” never equates class 0 with safety, and shows model-specific locked-test precision/recall, approximately 11% prevalence, imbalance/domain restrictions, and risks for reclaimed, intra-community, quoted, sarcastic, critical, or negated language. Clear resets only three input fields. No explanation/rewrite generation exists.

## 9. Environments and exact commands

Both local environments use CPython 3.14.0. `.venv` holds scikit-learn 1.9.1 and related baseline dependencies. `.venv-transformers` holds CPU PyTorch 2.14.1, Transformers 5.18.0, Streamlit 1.65.0 and supporting packages. Exact captured versions are in the two lock files; human-readable installation requirements remain unchanged.

```powershell
py -3.14 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-baselines.txt
py -3.14 -m venv .venv-transformers
& .\.venv-transformers\Scripts\python.exe -m pip install --prefer-binary --progress-bar off -r requirements.txt

& .\.venv-transformers\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
& .\.venv\Scripts\python.exe scripts\validate_partitions.py --config configs\partition.json
& .\.venv\Scripts\python.exe scripts\validate_baseline_run.py --config configs\baselines.json
& .\.venv-transformers\Scripts\python.exe scripts\validate_transformer_run.py --config configs\transformer_training.json
& .\.venv-transformers\Scripts\python.exe scripts\validate_final_test_run.py --config configs\final_test_evaluation.json
& .\.venv-transformers\Scripts\python.exe scripts\validate_v1_1.py --run-synthetic-inference --finalize
& .\.venv-transformers\Scripts\python.exe -m streamlit run app.py
```

Never run the Phase 6 executor. The old `validate_final_project.py` checks byte identity of the Phase 7 package, including UI/source files intentionally superseded by V1.1, and is therefore a historical validator after this release. See `reports/REPRODUCIBILITY_GUIDE.md` for clean-room and CPU-only details.

## 10. Important files and authority order

1. Immutable executable code/config/artifacts and their hashes.
2. `results/final_test/test_access_receipt.json`, final run manifest/status, and active validators.
3. `results/frozen_policy_v1_1/frozen_policy.json` (canonical policy identifier `b5765e9431a4bf01e8abfeeec1762061b1c0a26eec4ea26c326abb7245ac7541`).
4. `results/v1_1_validation/pre_change_integrity.json` and `post_change_integrity.json`.
5. V1.1 JSON audits and this handoff.
6. Historical reports, interpreted with the additive erratum.

Core implementation locations: loss and token building in `scripts/train_transformers.py`; calibration in `scripts/run_calibration_review.py`; final metrics/bootstrap in `scripts/evaluate_locked_test.py`; formatter in `src/sandarbh_ui/result_formatter.py`; inference in `src/sandarbh_ui/inference.py`; UI in `app.py`.

## 11. Immutable components

Treat AUTALIC, audit/overlap/partition artifacts, E1-E7 checkpoints, transformer unit outputs, calibration parameters/predictions, both frozen policies, all final-test artifacts, and historical scientific reports as immutable Version 1 evidence. V1.1 integrity manifests enumerate 130 files. Do not delete negative-test fixtures or historical validators merely because they are not current release gates.

## 12. Known issues and unknowns

- Official AUTALIC byte identity, local retrieval URL/date: **UNKNOWN / NEEDS VERIFICATION**.
- Cause/status of 122 empty rows and paper 242 versus local 256 positives: **UNKNOWN / NEEDS AUTHOR VERIFICATION**.
- Detailed locked-test token truncation counts: **UNAVAILABLE**; only maximum length and E6/E7 identity were saved.
- E7 soft-label causal effect: unidentifiable from E6/E7 because of weighting confound.
- Probability skill over train prevalence: not demonstrated under binary Brier.
- Generalization outside this dataset/domain and consequential use: not established.
- Additional paired intervals requested post hoc: unavailable under the frozen bootstrap and not generated.

## 13. Future work

Version 2 must start with a new protocol. Highest priorities are the hard/soft × weighted/unweighted 2×2 ablation, training-seed-aware uncertainty, broader leakage controls, independent/community-led external validation, controlled context reasoning, missing-context analysis, author provenance resolution, and governance before any consequential application. Details are in `reports/DEFERRED_TO_VERSION_2.md`.

## 14. Release and Git

The V1.1 package contains only the thirteen requested text-free handoff files plus `PACKAGE_CONTENTS.md`; each copy is byte-identical to its project original. `.gitignore` excludes data, results, models, environments, caches, logs, secrets, and prompt packages. CI runs artifact-independent synthetic UI tests only and downloads neither AUTALIC nor checkpoints.

Do not commit/tag/push automatically. After human review, inspect `git status` and `git diff`, then stage only intended source/docs/tests/config/CI files. Generated private outputs remain ignored. A suggested commit/tag sequence appears in the final agent report; tagging must follow explicit approval.

## 15. V1.1 validation record and test-count map

The complete discovery command belongs to `.venv`, which has baseline dependencies: 240 tests passed with 6 expected skips for PyTorch/Streamlit-only cases. The focused `.venv-transformers` command ran 30 V1.1/UI tests with no skips or failures. Running complete discovery in `.venv-transformers` is not the correct all-project command because that intentionally isolated environment lacks joblib; the attempted command ran 227 tests and reported one baseline-module import error. This is an environment separation issue, not a scientific failure.

Active saved-artifact validators passed: partitions 20 checks, baselines 17, transformer preflight 14, transformer training 17, calibration review 30, and final test 42. The V1.1 validator completed with 152 immutable files identical pre/post and 33 authored synthetic inference paths (11 cases × E5/E6/E7). The bounded Streamlit server returned HTTP 200 `ok` and was terminated. Both `pip check` commands, Python compilation, package hash verification, and `git diff --check` passed. The historical Phase 7 final-project validator passed 38/39 and failed only its old-package byte-identity check because V1.1 intentionally changes UI/source files; its generated summary was restored to the immutable historical copy.
