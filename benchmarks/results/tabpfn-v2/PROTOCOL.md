# TabPFN v2 benchmark protocol — 2026-09-22

Declared before the main benchmark. This is a small-sample comparison, not full-scale data performance or a superiority guarantee.

- Datasets: cancer, diabetes, adult, california, covtype, online_retail. Keep original targets; no learned pre-split clipping.
- Seeds: 11, 22, 33, 44, 55. Each seed uses at most 1,200 rows (stratified for classification), then a paired 25% untouched outer holdout and 3-fold inner CV. CPU refit has at most 900 rows. CoverType is seven-class classification. Online Retail is an IID transaction task, not a temporal/group-disjoint forecasting benchmark.
- Fixed downstream model: TabPFN v2, tabpfn 9.0.0, ensemble size 4, CPU. Locked weights and revisions are defined in [the model adapter](../../../featune/tabpfn.py). No parameter tuning. V2 classifier checkpoint is upstream-finetuned: public-data pretraining overlap cannot be excluded.
- Methods: raw baseline, Featune random, evolution, semantic LLM, anonymous LLM, OpenFE row-local. Four attempted search trials for Featune, at most one new feature each. Same data values/splits for semantic ablation; anonymous LLM hides names/descriptions/objective. No retrying seeds to improve scores.
- OpenFE: official 0.0.12, fixed seeded 32 row-local candidates, at most 10 selected outputs; internal LightGBM selection followed by the identical TabPFN. This is neither full OpenFE nor a fit-budget-matched comparison.
- Additional original-feature baselines: LogisticRegression/Ridge and HistGradientBoosting on the same row subsets and holdouts in the same environment.
- LLM: configured claude-opus-4-8 via Anthropic-compatible endpoint, temperature 0, max output 2,048 tokens per request, per-run budget 40,000 tokens. Record failed/unknown-billing calls; monetary cost is unknown unless prices are supplied. Never save credentials.
- Accuracy: outer AUC (weighted OVR for multiclass), accuracy; regression RMSE, MAE, R2. Report paired deltas vs raw TabPFN and semantic-vs-anonymous deltas. Five-seed bootstrap 95% intervals are descriptive and unadjusted for multiple comparisons, not broad statistical proof.
- Efficiency: wall time, CV fit count, candidate evaluations, tokens, valid/failed attempts, time to inner gain. Warm checkpoint/import costs once before method timing. Explanation and secondary-metric timing are separate. Do not compare old environments or historical full-data scores.
- Explanations: preserve expressions, hypotheses, lineage and inner-CV trial evidence. Disable in-search permutation equally for all Featune methods; do not claim causal/hypothesis confirmation from missing permutation evidence. After freezing the model, permute only selected compiled derived columns on outer holdout (3 repeats), with no feedback to selection. This measures predictive reliance, not causal importance; descendants are not recomputed.
- Preserve every run/failure. Preflight smoke runs are excluded. Use only completed paired runs and disclose missing results. Store dataset split hashes and version manifests.

## Repository retention — 2026-09-23

The protocol above governed execution. After aggregation, raw JSON/CSV and per-run directories were removed at the project owner’s request. Git retains the report, aggregate Markdown tables, this protocol, environment notes and the comparison figure. The published summaries cannot regenerate arbitrary per-run statistics. Re-run under ignored `runs/` to produce new local audit artifacts; this retention change does not change the recorded experimental outcomes.
