# SANDARBH Phase 4 Transformer Plan

## Phase 4A boundary

Phase 4A establishes a separate transformer environment, resolves an immutable DistilRoBERTa revision, measures real token lengths on `train` and `dev_tune`, and runs a bounded synthetic optimizer benchmark. It does not fine-tune on AUTALIC, generate real model predictions, inspect model-facing `dev_calibration` or `test` content, calibrate, or evaluate the locked test set.

The tokenizer and model are fixed to `distilbert/distilroberta-base` revision `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`, with `trust_remote_code=False`, two output labels, and safetensors weights. A newly initialized classification head is expected because the base checkpoint is not a trained SANDARBH classifier.

## Planned later experiments

| ID | Input | Target |
|---|---|---|
| E5 | target only | hard binary label |
| E6 | target with context | hard binary label |
| E7 | identical context construction to E6 | soft binary target |

Phase 4A does not execute E5–E7. Hyperparameters, early stopping, and checkpoint selection remain to be frozen in the next protocol before real fine-tuning.

## Proposed target-preserving construction

The proposed maximum encoded length is 256 tokens, subject to the Phase 4A diagnostics. Normalize each field with `" ".join(text.split())` while preserving case. Tokenize field markers and field content separately without special tokens, then add the tokenizer's outer special tokens and count them inside the 256-token budget.

Use the ordered segments `Preceding context:\n`, preceding content, `Target:\n`, target content, `Following context:\n`, following content. Always budget all three markers, even for missing context. Missing content contributes zero tokens. The same target tokenization and cap apply to target-only and context examples. If a target itself is too long, retain its leftmost token prefix after reserving all marker and outer-special tokens; record this explicitly rather than allowing ordinary right truncation to remove it silently.

After preserving the target, divide remaining context capacity equally. Keep the tail of preceding context and the head of following context so retained context is nearest the target. If one side is short or missing, deterministically give its unused capacity to the other side. E6 and E7 must receive byte-for-byte identical input IDs and attention masks for each example. Target-only E5 uses the same retained target tokens, surrounded by its target marker and outer special tokens, leaving any context-marker reservation unused.

This is a recommendation for the next phase, not an approved or executed training policy. Labels and development performance are not used to choose the sequence budget.

## Benchmark interpretation

The synthetic benchmark uses fixed neutral text, full-model gradients, AdamW state, FP32, length 256, and microbatch 1 before microbatch 2. It permits at most one warm-up and three measured optimizer steps per configuration and has explicit timeouts. Its epoch projection is only a linear extrapolation from measured optimizer-step time. Validation, checkpointing, variable sequence lengths, dataloader work, and competing hardware load can change actual runtime.

CPU success establishes functional feasibility, not practical training speed. CUDA usability is determined by `torch.cuda.is_available()`, not by the mere presence of a display adapter.
