# SANDARBH Version 1 research questions

Version 1.1.0 — 2026-10-07

## Primary frozen questions

1. Under the prespecified group-aware locked-test evaluation, how do the seven frozen experiments perform on macro-F1 and positive-class behavior?
2. For the transformer comparisons, what are the paired fixed-model differences for E6-E5 (context versus target-only hard-label training) and E7-E6 (cost-weighted soft-vote versus hard-label contextual training) under the frozen cluster bootstrap?
3. What are the frozen models' discrimination, probability, and selective-prediction summaries using the prespecified metrics and 0.60 operating point?

The questions were frozen before the one-time test. Results are descriptive; E2's numerically highest macro-F1 is not a proven winner. E7-E6 is not a clean soft-label causal contrast because E7's class-weighted soft objective changes the target optimum.

## Secondary/descriptive comparisons

Baseline-versus-transformer tables, per-class precision/recall, calibration before/after temperature scaling, coverage/risk curves, and seed-42/43/44 development variation are secondary descriptive evidence. Context altered operating behavior, but does not demonstrate correct contextual reasoning.

## V1.1 post-hoc questions

All items below are **POST-HOC / EXPLORATORY / NOT USED FOR MODEL SELECTION**:

- Does probability scoring beat a train-prevalence constant reference?
- How do saved errors/confidence vary by agreement, entropy, context availability, and overlap-group size?
- How often do E5/E6/E7 disagree?
- How are errors distributed between accepted and abstained examples at 0.60?
- Is development prediction uncertainty monotonically associated with vote entropy?

These questions cannot be retroactively treated as primary or used to choose a new model or threshold.
