# Phase 04A Transformer Preflight

**Status:** `COMPLETED`  
**Generated (UTC):** 2026-10-05T09:18:53.616416Z  

Phase 4A loaded the fixed DistilRoBERTa tokenizer and binary classification model, tokenized only `train` and `dev_tune` for diagnostics, and ran a synthetic optimizer benchmark. It did not fine-tune on AUTALIC or generate real predictions.

## Environment

- Device: `cpu`; `torch.cuda.is_available()` = `false`.
- CPU: Intel(R) Core(TM) Ultra 5 125H (18 logical processors).
- RAM: 16573128704 bytes total; 1078607872 bytes available at inspection.
- Model revision: `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`.

## Token lengths

All values are actual tokenizer counts. Final lengths include special tokens.

| Partition | Field | Median before/final | P95 before/final | Max before/final | >256 | >512 |
|---|---|---:|---:|---:|---:|---:|
| train | target | 23.0/25.0 | 60.0/62.0 | 399/401 | 2 | 0 |
| train | preceding | 15.0/17.0 | 53.0/55.0 | 419/421 | 1 | 0 |
| train | following | 17.0/19.0 | 52.0/54.0 | 610/612 | 2 | 1 |
| train | context | 61.0/63.0 | 143.3/145.3 | 1201/1203 | 6 | 2 |
| dev_tune | target | 24.0/26.0 | 51.0/53.0 | 122/124 | 0 | 0 |
| dev_tune | preceding | 15.0/17.0 | 45.5/47.5 | 208/210 | 0 | 0 |
| dev_tune | following | 18.0/20.0 | 45.0/47.0 | 81/83 | 0 | 0 |
| dev_tune | context | 59.0/61.0 | 112.5/114.5 | 369/371 | 1 | 0 |

## Ordinary right-truncation effect on target

| Partition | Limit | Partial target removal | Complete target removal |
|---|---:|---:|---:|
| train | 256 | 2 | 1 |
| train | 512 | 1 | 0 |
| dev_tune | 256 | 1 | 0 |
| dev_tune | 512 | 0 | 0 |

## Synthetic benchmark

- Batch 1, length 256, FP32: 2.244 seconds/step across 3 measured steps (`cpu`).
- Batch 2, length 256, FP32: 3.234 seconds/step across 3 measured steps (`cpu`).
- Three-epoch linear extrapolation: 7741.3 seconds = ceil(1595/2) × 3 × 3.234. This excludes validation, checkpointing, variable lengths, data loading, and hardware contention.

## Recommendation

Use a proposed maximum length of **256** with explicit role markers, target-first budgeting, preceding-tail retention, and following-head retention. Overlong targets retain the declared prefix cap. E6 and E7 must have identical input IDs and masks. This is a proposed next-phase policy, not an approved or executed training setup.

## Boundaries

No real fine-tuning, real classification prediction, calibration, threshold selection, transformer checkpoint saving, `dev_calibration` tokenization, or `test` tokenization/evaluation occurred. The synthetic optimizer state and updated weights were discarded.
