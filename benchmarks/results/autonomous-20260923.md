# Autonomous search and context checks · 2026-09-23

These are new, preliminary experiments. The old six-dataset v2 results were not reused for model or prompt selection. Raw run records remain under ignored `runs/`; this file preserves the protocol and aggregate observations.

## Independent search comparison

The downstream model was multinomial logistic regression. Each method used the same outer 25% holdout and two inner CV folds for each seed. Random and autonomous search had the same trial count, maximum of three derived columns and model-fit ceiling within each run. Autonomous screening and duplicates can use fewer fits; the ceiling is equal, actual fits are reported.

| Dataset / budget | Seeds | Method | Mean outer weighted OVR AUC | Mean actual fits incl. refit | Mean selected columns |
|---|---:|---|---:|---:|---:|
| Wine / 4 attempts, fit cap 11 | 101, 202 | baseline | 0.999252 | 3.0 | 0.0 |
| Wine / 4 attempts, fit cap 11 | 101, 202 | random | 0.999252 | 11.0 | 0.0 |
| Wine / 4 attempts, fit cap 11 | 101, 202 | autonomous | 0.999252 | 6.0 | 0.0 |
| Digits / 4 attempts, fit cap 11 | 101, 202, 303 | baseline | 0.998008 | 3.0 | 0.0 |
| Digits / 4 attempts, fit cap 11 | 101, 202, 303 | random | 0.997926 | 11.0 | 1.33 |
| Digits / 4 attempts, fit cap 11 | 101, 202, 303 | autonomous | 0.997877 | 6.33 | 0.67 |
| Digits / 8 attempts, fit cap 19 | 101, 202, 303 | baseline | 0.998008 | 3.0 | 0.0 |
| Digits / 8 attempts, fit cap 19 | 101, 202, 303 | random | 0.997926 | 19.0 | 1.67 |
| Digits / 8 attempts, fit cap 19 | 101, 202, 303 | autonomous | 0.997959 | 10.33 | 1.67 |

Digits used 600 presplit rows. At eight attempts, random and autonomous happened to select the same number of columns for each paired seed (1, 1, 3). Their outer scores were close, and both means were below the raw baseline. The three seeds do not support a general superiority claim. The four-attempt arm shows the effect of a smaller budget; its selected column counts differ, so it is not an exact feature-count comparison. No current experiment forces an exact spent-fit count; that remains a stricter comparison to run before claiming equal-resource accuracy gains.

A separate 120-row, 120-column regression smoke completed six autonomous attempts, selected two derived columns and used 12 model fits. This establishes bounded wide-schema execution, not predictive benefit.

## Real LLM semantic audit

Three invented metadata-only tasks (credit, readmission and demand) were sent through the configured provider. The current context correctly identified downstream v2 model type and label meaning in all three. A metadata-ablation arm had no estimator or exact label definition and returned `unknown` on those fields. These are three cases, not a statistical misstatement rate. The first current prompt asserted that leakage was impossible in all three rationales; metadata cannot establish that. After the proposal instruction required timing assumptions and prohibited leakage certification, a separate three-case run again got all model/label claims right and all three rationales explicitly qualified their timing assumptions. This is a prompt-level observation; real feature availability still requires external verification.

## Real wide-context ablation

One seed (22), one LLM proposal per arm, Ridge regression, 200/500/1000 numeric columns, five context strategies. Each arm had a 60,000-token reservation ceiling. At 200 columns, all five arms returned valid proposals; actual input token counts were 21,964 (full), 12,594 (random), 10,951 (semantic), 12,704 (semantic grouping) and 2,672 (semantic memory). At 500 and 1000 columns, full context was stopped before an API request by the token ceiling; bounded strategies completed valid proposals. Outer RMSE changes were mixed and small. The single seed and synthetic task do not show a semantic or accuracy advantage.

Default TabPFN v2 real-weight classifier/regressor acceptance passed 12 tests with `FEATUNE_TEST_TABPFN=1`. Opt-in v3.5 validation was outside this run's scope.
