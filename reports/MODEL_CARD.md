# SANDARBH V1 model card

Version 1.1.0 addendum — 2026-10-07

## Intended use

SANDARBH is a research comparison of target-only and contextual ableism annotation models on AUTALIC. E1-E4 are frozen TF-IDF baselines; E5 is target-only DistilRoBERTa with hard labels; E6 is contextual DistilRoBERTa with hard labels; E7 uses the identical contextual input and a cost-weighted soft-vote loss. The Streamlit application is a local research demonstration.

Not intended for clinical use, diagnosis, automated moderation, punishment, employment, education, access decisions, or claims that text or a person is safe, harmful, ableist, or non-ableist. Automated moderation requires prior dataset-author approval in addition to technical validation not supplied by V1.

## Inputs and outputs

Inputs are a required target plus optional preceding/following context. E5 ignores context; E6/E7 use it. Outputs are a predicted benchmark class, temperature-scaled model-estimated probabilities, confidence, and whether confidence is above the frozen 0.60 research threshold. Class 0 means: “No majority ableist annotation under this benchmark's binary mapping. This does not prove that the text is harmless or acceptable.”

## Training and frozen state

The 2,278 eligible records are group-partitioned into train 1,595, dev-tune 171, dev-calibration 171, and locked test 341. The one-time test is complete. Seed 42 checkpoints and epochs are frozen: E5 epoch 3, E6 epoch 2, E7 epoch 2. Temperatures are E5 1.0237168508133028, E6 0.6020567132971402, E7 0.6154062772305571. The decision boundary is 0.50; selective confidence threshold is 0.60.

## Locked-test context

Positive prevalence is 38/341 (11.14%). Positive precision/recall are E5 0.270/0.711, E6 0.315/0.447, and E7 0.255/0.711. E2 has the highest observed macro-F1 (0.653907), but V1 does not establish a winner. Every available probability-producing model has negative Brier skill versus the train-prevalence constant reference.

## Limitations and risks

AUTALIC is domain- and culture-specific; label agreement is limited; class imbalance is severe. Intra-community, reclaimed, quoted, sarcastic, critical, negated, or identity-referencing language can be misclassified. Context changes predictions but does not prove contextual understanding. Exact-overlap grouping does not prevent every semantic leakage route. Confidence and abstention are not safety guarantees. Bootstrap intervals omit training-seed variability. E7's weighting shifts the effective soft-target optimum, so E7-versus-E6 is confounded.

## Reproducibility and provenance

See `PROJECT_HANDOFF_V1_1.md`, `reports/REPRODUCIBILITY_GUIDE.md`, and `results/v1_1_validation`. Local checkpoint and data placement are required; neither is redistributed in the handoff package.
