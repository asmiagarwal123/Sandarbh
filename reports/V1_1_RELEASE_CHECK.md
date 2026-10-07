# SANDARBH V1.1 release-consistency check

Date: 2026-10-07  
Scope: live repository UI wording, focused tests, current V1.1 validator, and frozen-artifact integrity only.

## Files inspected

- `app.py`
- `src/sandarbh_ui/inference.py`
- `src/sandarbh_ui/result_formatter.py`
- `tests/test_ui.py`
- `tests/test_v1_1_validation.py`
- `reports/MODEL_CARD.md`
- `PROJECT_HANDOFF_V1_1.md`

All inspections used the live project files, not prompt-package copies.

## Wording result

Wording-only changes were required to normalize the active E7 UI label to the exact phrase:

> Contextual cost-weighted soft-vote training

The earlier wording was scientifically accurate but did not use that exact contiguous label in every active UI location: the selector metadata used `Contextual; cost-weighted soft-vote training`, and the model-information panel referred to a `cost-weighted soft-vote objective`.

The live UI now states the exact phrase in both locations. A focused regression test requires the phrase in `MODEL_SPECS["E7"]["description"]` and `app.py`.

Verified interpretation:

- E5 remains target-only hard-label training.
- E6 remains contextual hard-label training.
- E7 remains contextual cost-weighted soft-vote training.
- E7-versus-E6 is explicitly described as confounded and not a clean causal soft-label comparison.
- E7 is not presented as directly learning raw annotator-vote probabilities or as ordinary unweighted soft cross-entropy.
- Class 0 is not described as safe or unconditionally non-ableist.
- The frozen 0.60 value is described only as a research confidence/review operating point, not a safety guarantee.
- No model is presented as a proven winner.
- The dashboard remains a research demonstration and explicitly not a moderation system.

## Exact files changed

- `app.py` — wording only; exact E7 label and explicit confounding sentence.
- `src/sandarbh_ui/inference.py` — wording only in the E7 description; no path, hash, mode, temperature, inference logic, or numerical behavior changed.
- `tests/test_ui.py` — added one focused exact-phrase regression test.
- `reports/V1_1_RELEASE_CHECK.md` — this report.

No wording change was required in `result_formatter.py`, `test_v1_1_validation.py`, `MODEL_CARD.md`, or `PROJECT_HANDOFF_V1_1.md`; their scientific and safety descriptions were already accurate.

## Tests and validators

Executed with `.venv-transformers`:

- Python compilation of `app.py`, UI modules, focused tests, and `validate_v1_1.py`: passed, exit 0.
- `python -m unittest tests.test_ui -v`: 19 tests passed, exit 0.
- `python -m unittest tests.test_v1_1_validation -v`: 12 tests passed, exit 0.
- `scripts/validate_v1_1.py --config configs/v1_1_validation.json --run-synthetic-inference --finalize`: `COMPLETED`, exit 0.
- Synthetic authored model smoke paths: passed. No locked-test inference was run.
- `git diff --check`: passed, exit 0.

## Immutable-artifact integrity

The V1.1 validator checked 152 frozen scientific files.

- Pre-change integrity status: `PASS`
- Post-change integrity status: `PASS`
- Pre/post path and SHA-256 lists identical: `true`
- Scientific artifacts unchanged: `true`
- Locked-test source text loaded: `false`
- Locked-test inference rerun: `false`
- Version 2 experiment performed: `false`

No prediction, checkpoint, partition, temperature, threshold, policy, metric, bootstrap output, or other immutable scientific artifact changed.

## Final status

`READY_TO_TAG`

No commit, tag, or push was performed.
