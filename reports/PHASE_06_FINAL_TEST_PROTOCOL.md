# SANDARBH Phase 6 Final Locked-Test Protocol

Phase 6 is the first and only evaluation of the 341-example locked test partition. It is governed exclusively by successor policy SHA-256 `b5765e9431a4bf01e8abfeeec1762061b1c0a26eec4ea26c326abb7245ac7541`; the superseded policy remains preserved but does not control this evaluation.

The evaluator has two mutually exclusive modes. `--preflight-only` validates statuses, canonical policy identity, frozen data provenance, structural test counts, group isolation, artifact absence, and model hashes without loading test text or performing vectorization, tokenization, or inference. `--execute-once` atomically creates a permanent access receipt immediately before test loading. A completed run can never be executed again; an interrupted run may only recover with identical code, configuration, policy, data, and model provenance.

E1–E4 use their serialized Phase 3 pipelines without fitting. E1/E2 expose native positive-class probabilities; E3/E4 expose signed decision margins, never probabilities. DUMMY always predicts label 0 and has no continuous score. E5–E7 use the seed-42 checkpoints, selected epochs 3/2/2, exact Phase 4B token construction, CPU FP32 inference, and frozen temperatures 1.0237168508133028, 0.6020567132971402, and 0.6154062772305571. Temperature scaling must preserve argmax.

All systems receive exactly the same IDs. The analysis reports frozen classification and ranking metrics, transformer probability quality before and after calibration, E1/E2 native probability quality, 10-bin reliability data, the complete 0.50–0.99 selective grid with 0.60 as the frozen operating point, six declared comparisons, and 5,000 paired Rule-B-group bootstrap replicates using seed 20261006 and percentile 95% intervals. Unavailable values remain unavailable; no p-values or resampling-until-valid are allowed.

No training, refitting, recalibration, model selection, threshold change, temperature change, or metric addition is allowed after access. Results are descriptive for AUTALIC and this split. Abstention is not correctness, low selective risk is not safety, label 0 is not a definitive safe judgment, soft votes are not ground-truth probabilities, and no causal or deployment-readiness claim is permitted.

