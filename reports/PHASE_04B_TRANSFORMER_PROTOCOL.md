# SANDARBH Phase 4B Transformer Training Protocol

## Scope and fixed decisions

Phase 4B runs nine sequential DistilRoBERTa development trainings: E5 target-only hard labels, E6 target-with-context hard labels, and E7 the identical context input with soft binary targets, each with seeds 42, 43, and 44. Seed 42 is predesignated for later calibration/final stages. Seeds 43 and 44 are robustness runs and can never replace seed 42 based on development performance.

The model is `distilbert/distilroberta-base` at immutable revision `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`, safetensors only, `trust_remote_code=False`. Training is CPU FP32, three epochs, microbatch 2, gradient accumulation 8, effective batch 16, AdamW learning rate `2e-5`, weight decay `0.01`, clipping norm `1.0`, and linear decay after a 10% warmup. No parameters are frozen, gradient checkpointing is off, and every run completes all three epochs.

Only `train` is used to fit weights. Only `dev_tune` is evaluated after each epoch and used to select a checkpoint. The loader rejects `dev_calibration` and `test`. Phase 4B performs no calibration, threshold tuning, selective prediction, final evaluation, ensembling, hyperparameter search, or training on combined partitions.

## Exact input construction

Each text field is normalized with `" ".join(text.split())`; case, punctuation, and wording are preserved. The tokenizer handles field content and these exact marker strings without automatic special tokens:

```text
Preceding context:\n
Target:\n
Following context:\n
```

The final sequence is assembled from token IDs and receives exactly one beginning token and one ending token. Every special and marker token counts toward the fixed 256-token limit. The expected marker lengths are six, three, and four tokens; after reserving those thirteen marker tokens and two outer tokens, the fixed target-content cap is 241. Execution must recalculate and fail if this differs.

All experiments tokenize target content identically. A target of at most 241 tokens is retained completely. For an overlong target, tokenize the literal ` …`, subtract its token count from 241, give the beginning the extra content token when the remaining capacity is odd, and retain `beginning + ellipsis + ending`. This replaces Phase 4A's provisional prefix-only rule with the approved Phase 4B refinement.

E5 is `BOS + target marker + retained target + EOS`. It uses the same 241-token target cap despite unused capacity. E6 and E7 are `BOS + preceding marker + preceding content + target marker + retained target + following marker + following content + EOS`. Remaining content capacity is split equally, with the odd token assigned to preceding. Preceding retains its tail and following retains its head. All unused capacity on a short or missing side is transferred deterministically to the other; markers remain even when context is missing. There is no second truncation operation.

E6 and E7 consume the same stored context encoding. Their complete train/dev encoded-manifest hashes must therefore be identical. The text-free tokenization manifest records stable IDs, original and retained lengths, truncation flags, and cryptographic hashes, never text or decoded tokens.

## Targets and losses

Class weights are derived once from the frozen train hard labels as `N / (2 * count_c)` and shared across E5–E7. E5/E6 use the mean per-example weighted hard cross-entropy. E7 uses the mean weighted soft cross-entropy:

```text
-sum_c weight_c * soft_target_c * log_softmax(logits)_c
```

The hard implementation is the one-hot special case of the soft implementation. The soft target remains `[soft_other, soft_positive]`; weights do not trigger per-example renormalization. Original `-1`, `0`, and `1` annotations remain unchanged in the frozen target artifact and are not modeled as a third class.

## Optimization and checkpoint selection

Training order is shuffled deterministically for each seed and epoch. Python, NumPy, and PyTorch seeds are set before model initialization. CPU deterministic algorithms are requested with warning-only handling for unsupported operations. Dynamic padding uses the longest sequence in each microbatch and never exceeds 256.

The scheduler counts optimizer updates, not microbatches. With 1,595 examples, microbatch 2 produces 798 microbatches and `ceil(798/8)=100` optimizer updates per epoch. The last partial accumulation group is scaled by its actual number of microbatches and is stepped rather than discarded. Total updates are 300 with 30 warmup updates.

After every epoch, the model is evaluated only on `dev_tune`. Each experiment-seed selects the highest exact unrounded macro-F1 over labels `[0,1]`; ties select the earliest epoch. Predictions use native argmax and no tuned threshold. Only the three selected seed-42 checkpoints are retained locally. Robustness weights for seeds 43 and 44 are absent after their metrics and predictions are atomically published.

## Metrics and interpretation

Every epoch records macro-F1, positive precision/recall/F1, accuracy, balanced accuracy, `[0,1]` confusion matrix and support, average precision, ROC-AUC, predicted-positive count, and the no-positive flag. Selected checkpoints additionally record hard-label NLL, Brier score, ten-bin equal-width ECE, soft cross-entropy, soft Brier score, and mean absolute difference from the annotator-derived `soft_positive` fraction.

All values are `dev_tune` development estimates. ECE is descriptive and unstable with only nineteen hard positives. Agreement with three votes is not ground-truth probability calibration. Across-seed summaries use mean, sample standard deviation, minimum, and maximum. Paired per-seed E5−E6 and E6−E7 differences are descriptive; no significance test, overall winner, or final generalization claim is permitted.

## Recovery and validation

Each experiment-seed is an atomic resumable unit keyed by experiment, seed, configuration/code/input hashes, frozen source/partition provenance, and model revision. Fully completed identical units are independently checked and reused. Corrupt or changed-provenance units are rejected. Incomplete work restarts from the pretrained initialization. Aggregates are republished after each unit so completed work survives interruption.

The independent validator is read-only. It reconstructs provenance, token retention, metrics, selection, summaries, and comparisons; verifies exactly nine three-epoch units; checks seed-42 checkpoint hashes before local loading; confirms robustness weights are absent; and confirms no `dev_calibration` or `test` access. Previous source, partitions, baselines, and Phase 4A artifacts must remain byte-identical.
