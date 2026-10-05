# Phase 02A Overlap-Group Review

**Policy status:** `PENDING_RESEARCHER_REVIEW`  
**Generated (UTC):** 2026-10-04T10:47:21.057825Z  

This report compares deterministic proxy grouping rules for leakage control. It does not approve a grouping policy or create a data split.

## Candidate rules

Rule A connects identical nonempty normalized targets regardless of length and also connects examples sharing any normalized text field of at least five words. Rule B connects examples sharing any normalized nonempty text field regardless of length. Both use connected components and retain every unconnected eligible example as a singleton.

Normalization is `" ".join(text.split()).casefold()` for matching only. It does not establish model-input preprocessing. Labels do not create edges.

| Measure | Rule A | Rule B |
|---|---:|---:|
| Eligible examples | 2278 | 2278 |
| Total groups | 1462 | 1425 |
| Singleton groups | 1020 | 988 |
| Nontrivial groups | 442 | 437 |
| Examples in nontrivial groups | 1258 | 1290 |
| Largest group | 17 | 22 |
| Diagnostic binary-positive examples | 256 | 256 |
| Groups containing a diagnostic binary-positive example | 202 | 201 |

The binary mapping remains descriptive only: two or more votes of `1` are positive; votes of `0` or `-1` map to the other category. It does not finalize training targets.

## What changes under Rule B

Rule B merges 24 Rule A group combinations and affects 122 examples. 25 distinct short normalized strings create additional connections. Their word-count distribution is `{"1": 8, "2": 6, "3": 2, "4": 9}`.

The repeated-target exception changes the five-word-only components for 0 target examples across 0 repeated normalized target strings.

These groups are proxies: exact text reuse can indicate leakage risk, but short generic phrases can connect unrelated passages, while exact matching cannot recover source-post identity or paraphrases.

## Consistency and reference observations

- all_eligible_examples_have_rule_a_membership: True
- all_eligible_examples_have_rule_b_membership: True
- all_normalized_duplicate_targets_grouped_by_rule_a: True
- all_normalized_duplicate_targets_grouped_by_rule_b: True
- all_substantial_exact_text_links_within_rule_a: True
- all_substantial_exact_text_links_within_rule_b: True
- every_rule_a_group_contained_in_one_rule_b_group: True
- All configured independent reference observations match their calculated definitions.

## Human review

Review `results/overlap_review/manual_review.csv`. It contains no raw text. Locate a source record using `source_row`, the one-based logical CSV data record excluding the header. In a correctly imported spreadsheet, the displayed row is normally `source_row + 1`; do not treat it as a physical text-editor line number because CSV fields may contain embedded newlines.

Use review only to understand grouping-policy behavior. Do not make ad hoc label changes, delete examples, or add metric-driven exceptions. Record observations and notes in the blank reviewer columns; reruns preserve notes by stable review case ID.

## Pending decisions

- Choose between Rule A and Rule B, or explicitly authorize a later alternative analysis.
- Decide whether short generic exact matches represent unacceptable leakage risk.
- Confirm how candidate components will constrain a future partitioning procedure.

No final split or grouping policy has been approved. No train/validation/test membership, training, calibration, or model evaluation was produced.

## Warnings and failures

- No warnings.
- No blocking failures.
