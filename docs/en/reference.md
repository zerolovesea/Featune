# DSL and API reference

[中文](../zh/reference.md) · [Guide](guide.md)

## Controlled DSL

```json
{
  "features": [{
    "name": "debt_income_ratio", "op": "divide", "inputs": ["loan", "income"], "params": {},
    "hypothesis": {
      "concept": "repayment_capacity",
      "rationale": "Principal relative to monthly disposable income approximates repayment burden",
      "expected_direction": "increasing"
    }
  }],
  "remove": [],
  "model_params": {"C": 1.0}
}
```

`remove` names parent derived features to delete; their descendants must also be deleted. Model parameters must match both the value and type in the study's approved param_space; otherwise use `{}`. Feature names start with an ASCII letter, contain only ASCII letters/digits/underscores and have at most 100 characters. Original field names may contain spaces or non-ASCII text. Every feature needs a concept and rationale; expected direction is increasing/decreasing/unknown. Inputs must be usable raw fields or earlier generated features. Duplicate expressions, name collisions, forward references and extra schema keys are rejected.

| Operation | Inputs | Definition |
|---|---|---|
| add / subtract / multiply | numeric, numeric | a+b / a−b / a×b |
| divide | numeric, numeric | a/b; missing when abs(b) ≤ 1e-12 |
| log1p_abs / sqrt_abs | numeric | log(1+abs(x)) / sqrt(abs(x)) |
| square / abs | numeric | x² / abs(x) |
| clip | numeric | Exactly lower and upper parameters, lower < upper |
| is_missing | any | 1 if missing, else 0 |
| frequency | categorical | Training relative frequency; unseen values map to 0 |
| cross | categorical, categorical | Length-prefixed category pair |
| group_mean | categorical, numeric | Training group mean; unseen groups use global training mean |
| group_std | categorical, numeric | Training population std (ddof=0); unseen groups use global training std |
| year / month / dayofweek | datetime | Year / month 1–12 / weekday Monday=0 |
| days_between | datetime, datetime | (a−b) in seconds divided by 86400 |

All non-clip operations require empty params. Infinite outputs become missing and downstream fold-local preprocessing imputes them. Date missingness is evaluated before date-to-number conversion. List order determines dependency order.

## Study API

`create_study` returns FeatureStudy with these settings:

| Argument | Default | Meaning |
|---|---|---|
| metric / direction | auc / inferred | Direction follows the metric; contradictory explicit values are rejected |
| sampler | RandomSampler(42) | A BaseFeatureSampler |
| storage / study_name | None / default | Parent directory and single project directory name |
| search_strategy | greedy | independent, greedy, beam or autonomous |
| beam_width | 3 | Retained frontier size |
| max_features | 32 | Generated-feature limit, at most 64 |
| param_space | {} | Estimator parameter names mapped to nonempty finite choice lists |
| random_state | 42 | Default sampler/evaluator seed |

`optimize(X, y, schema, estimator=None, n_trials=30, evaluator=None, groups=None, timeout=None, patience=None, callbacks=(), catch=(ValueError, RuntimeError), refit=True)` returns the study.

An evaluator supplies its own estimator and must use the study's main metric. Candidate ValueError/RuntimeError failures become FAIL trials by default; LLMError is always recorded. Baseline failures propagate. Set `catch=()` to propagate original estimator exceptions while debugging. Persistence errors are never swallowed as model failures.

| Accessor / method | Result |
|---|---|
| best_trial / baseline_trial | Best COMPLETE Trial / baseline number 0 |
| best_features / best_value | list[FeatureSpec] / selected inner-CV mean |
| trials | Full history including failed and interrupted attempts |
| trials_dataframe() | One row per trial, including nested feedback |
| feature_history() | One row per trial-feature with hypothesis evidence |
| concept_summary() | Record counts by concept and status |
| stop() | Stop after the current trial |
| export_pipeline(path=None) | Refitted sklearn Pipeline; optionally atomic joblib export |
| export_features(path) | JSON of the selected Proposal |
| report(path=None) | Self-contained HTML; Plotly included by default |
| load_study(storage, study_name, sampler=None) | Load history; training data must still be supplied to resume |

`autonomous` defaults to `AutonomousSampler`: numerical proposals alternate with an optional LLM, and leaf features are periodically proposed for deletion. It adds at most one column per trial and continues trying deletions from the best set after reaching `max_features`. A column constant in every training fold becomes `SCREENED_OUT` without a model fit. Additions and deletions use the same complete CV folds. Eight consecutive non-improving attempts stop the search by default; the baseline remains eligible. Configure semantic proposals with `AutonomousSampler(llm_sampler=LLMSampler(features_per_trial=1))`. `max_features` is a safety ceiling.

Trial states are RUNNING, COMPLETE, FAIL, DUPLICATE, SCREENED_OUT, INTERRUPTED and BUDGET_EXCEEDED. Baseline is trial zero; interrupted numbers are not reused. Scores retain their natural direction. `delta=value−baseline` and `parent_delta=value−parent`, so negative deltas improve minimize tasks.

## Evaluation API

CVEvaluator accepts `estimator=None, metric="auc", cv=5, metrics=None, random_state=42, n_jobs=1, importance_repeats=None, importance_max_samples=500`. Set importance_repeats=0 to skip numerical explanation; positive-delta hypotheses without importance evidence remain inconclusive.

`prepare(X,y,groups)` fixes and checks folds. `evaluate(X,y,schema,features,model_params=None)` returns EvaluationResult. Call prepare again when directly reusing an evaluator on another dataset. Study handles preparation automatically.

| Metric | Direction | Definition |
|---|---|---|
| auc | maximize | Binary ROC AUC |
| auc_ovr | maximize | Weighted multiclass one-vs-rest AUC |
| accuracy | maximize | Accuracy |
| f1 | maximize | Weighted F1 |
| log_loss | minimize | Positive cross-entropy loss |
| rmse | minimize | Positive root mean squared error |
| mae | minimize | Positive mean absolute error |
| r2 | maximize | Coefficient of determination |

EvaluationResult contains value, metrics, fold_values, importance, directions and duration. Folds have equal weight in the mean and train independently.

## Hypothesis evidence

- retained: already present in the parent feature set.
- strongly_supported: the joint trial improves its parent, column importance is positive, at least 80% of folds improve, and expected direction is unknown or agrees with observation.
- supported: joint improvement, positive importance and no conflict with the expected direction, without satisfying all stronger conditions.
- rejected: the joint trial does not improve.
- inconclusive: joint improvement without sufficient importance evidence, or with conflicting observed direction.

Stability is high when at least 80% of folds improve, otherwise mixed. Direction is the mean held-out Spearman association between a numeric feature and numeric target, with absolute values at most 0.05 labeled unknown. These are heuristic search feedback, not significance tests, individual-feature ablations, causality or SHAP.

## LLM API

LLMSampler: `model, temperature=None, client=None, history_limit=8, max_context_chars=24000, repair_attempts=1, features_per_trial=8, context_builder=None, context_budget=None, retriever=None, retrieval_threshold=200, **client_kwargs`. Per-trial feature limits are enforced, from 1 to 32. Inject a client for custom transport or mock HTTP testing.

LLMClient: `model=None, provider="anthropic", base_url=None, api_key=None, temperature=None, max_tokens=2048, output_token_parameter=None, timeout=60, retries=2, token_budget=None, input_cost_per_million=None, output_cost_per_million=None, cost_budget=None, transport=None`. `temperature=None` omits that field. OpenAI defaults to `max_completion_tokens`; Anthropic and local OpenAI-compatible servers default to `max_tokens`. Legacy OpenAI-compatible gateways can set `output_token_parameter="max_tokens"`.

Unset model/base_url/key values read FEATUNE_MODEL, FEATUNE_BASE_URL and FEATUNE_API_KEY. Default model is claude-opus-4-8 and the default Anthropic host is api.anthropic.com. Explicitly choose a supported model when changing providers. `usage` exposes input_tokens, output_tokens, requests and unknown_requests; total_tokens sums known input/output usage.

## CLI

```bash
featune optimize examples/config.json
featune inspect --storage runs --study-name cli-demo
featune report --storage runs --study-name cli-demo --output runs/cli-demo/report.html
featune export --storage runs --study-name cli-demo --output runs/cli-demo/best-features.json
```

See the [complete configuration](../../examples/config.json). Paths resolve relative to the config file. Data may be CSV or Parquet (pyarrow required). Target is removed from input features; optional groups identifies the grouping column. Estimator types: tabpfn (default), tabpfn_classifier, tabpfn_regressor, logistic, ridge, hist_classifier, hist_regressor, random_forest, torch_classifier, torch_regressor. Samplers: random, evolution, llm, hybrid, autonomous; the autonomous `llm` sub-configuration is optional.

Rerunning optimize adds n_trials; it is not an idempotent read. Inspect/report/export are read-only. Export writes feature JSON; optimize writes the fitted model. Use `featune --verbose optimize ...` for verbose logging. Library logs use the `featune` logger so applications can install their own handlers.

## IR, FeatureSet and lineage

`FeatureSet.from_features(schema, features, source_trial=None, origins=None)` creates the canonical execution representation. Each `FeatureIR` contains the controlled FeatureSpec plus output_dtype, stateful, lineage_parents (input name → hash), generation, source_trial, content_hash and version. Its hypothesis carries the semantic concept. Hashes include operator, output name, parameters and dependency hashes; rationale/provenance do not change numerical identity. Independent input list order does not change the FeatureSet hash. The factory topologically sorts dependencies and rejects cycles; raw sampler proposals still require dependency order.

`FeatureSet.model_validate_json(feature_set.model_dump_json())` round-trips metadata. The compiler normalizes public FeatureSpec inputs through this representation before executing any operator. `study.lineage_dataframe()` exposes the DAG across trials. Each trial records parent_set, proposal, candidate_set, feature_count, parent_delta, fold_values, model_fits, transform_time and training_time.

## Lifetime budgets and compatibility

```python
budget = featune.SearchBudget(max_trials=20, max_model_fits=80,
                             max_wall_time=600, max_valid_proposals=15,
                             max_llm_tokens=100_000)
study = featune.create_study(budget=budget, joint_optimization=False,
                            include_statistics=False, cache=True)
```

All limits are optional and cumulative across resume. `n_trials` remains the additional attempts requested by one optimize call. Model fits include baseline folds, candidate folds and final full-training refit. A fit already running cannot be preempted: wall-time checks occur between fits/trials. Exhaustion stops normally and preserves history. If baseline cannot fit, there is no best_trial; if refit cannot fit, export_pipeline is unavailable. An already cached refit remains exportable without another fit. `study.consumption` reports lifetime usage.

`max_llm_cost` requires explicit `input_cost_per_million` and `output_cost_per_million`; otherwise cost is unknown (`None`), never zero. Prices are user-supplied estimates in one consistent currency, not a provider invoice; all reported input tokens, including cached input, use the input rate. Request preflight conservatively reserves UTF-8 input bytes plus maximum output. Unknown-billing requests block further budgeted requests. Different cache pricing requires a billing reconciliation outside these estimates.

`study.fingerprint` is an ExperimentFingerprint: package/source/DSL/compiler/prompt versions, git revision, Python and dependency versions, dataset/schema/CV/estimator hashes, sampler/provider/model and seed. Different identities reject resume. `optimize(..., allow_incompatible_resume=True)` archives old metadata/trials in SQLite, resets active accounting and starts a separate experiment; it does not mix incompatible results. Old cache namespaces remain isolated.

Six cache layers reuse IR, FeatureSets, fold transformers, transformed matrices, fitted estimators and evaluation results. Keys include the experiment namespace plus data/schema/folds/FeatureSet/estimator/compiler identity. Disk cache survives resume; in-memory heavy layers retain eight entries per layer. Set `cache=False` to disable artifacts; exact duplicate completed trials still reuse durable metrics. Cache and exported joblib files are trusted local artifacts and may contain training-derived matrices/statistics: protect their directory and never load untrusted joblib. SQLite and LLM prompts do not contain raw rows.

`include_statistics=True` sends target-free summaries from the training data supplied to optimize: dtype, missing rate, cardinality, numeric moments/quantiles and categorical frequencies without category labels. It is opt-in; do not pass outer-test rows to optimize. Joint estimator tuning requires `joint_optimization=True` plus an approved param_space.

`search_summary(target_gain=0.005)` reports candidate evaluations, model fits, tokens/cost, valid/invalid proposal rates, cache hit rate and resources to best/fixed gain/95% of final gain. Unreached targets are null. All attainment is based on inner CV; negative score deltas improve minimize metrics. `pareto_frontier(resource="evaluations")` also supports model_fits, wall_time, tokens and api_cost. Unknown costs cannot enter cost Pareto comparisons. Reports show these diagnostics, feature-set evolution and lineage.

`study.proposal_history()` and the report's “LLM / sampler proposals” table show the parsed proposal that passed or failed validation: feature operator, inputs, hypothesis concept/rationale, model parameters, state, error and token usage. Per-trial INFO logs also show validated hypotheses, assessed status/reason, parent delta, cache reuse and best-selection decision; each completed fold logs the primary metric. Raw model responses are neither persisted nor logged; the structured proposal is the executable and auditable artifact.

## Wide schemas and bounded context

```python
from featune import ContextBudget, ContextBuilder, SemanticColumnRetriever

context = ContextBuilder(
    budget=ContextBudget(max_columns=40, max_detailed_stats_columns=20,
                         max_history_trials=10, max_failed_patterns=10,
                         max_tokens=12000, max_concepts=6, max_lineage_items=20),
    retriever=SemanticColumnRetriever(top_k=40, concept_top_k=5, seed=42),
    retrieval_threshold=200,
)
# Pass context_builder=context to LLMSampler; no API call occurs while building it.
```

FieldSchema additionally accepts `semantic_tags` (up to 20 short strings), `entity`, `encoding` and `available_at`. DatasetSchema accepts optional `target_definition` and `prediction_point`. The latter two describe label semantics and when inputs must exist; they do not send target values. The prompt also identifies the actual estimator class and optional TabPFN model version. Field availability is descriptive, so post-prediction columns must still be excluded explicitly. Concepts use the first semantic tag, then partition, entity, or dtype. Partition descriptions participate in retrieval. The default local semantic index uses character TF-IDF over names/descriptions/tags/entity/unit/encoding/availability/dtype; it does not require remote embeddings and does not infer arbitrary synonyms. `BaseColumnRetriever` permits replacement; RandomColumnRetriever, RuleBasedColumnRetriever and HybridColumnRetriever are also provided. Hybrid reserves half its slots for seeded exploration. Fixed schema/history/seed/trial number yields the same selection.

Retrieval is automatic at `retrieval_threshold=200`, or sooner whenever all fields would exceed `max_columns`. Retrieval ranks concepts/columns before prompt construction. ContextBuilder applies the hard column/concept limits even to custom retrievers. Descriptions are compacted to 240 characters. Current parent names and dependencies remain available; generated proposals cannot reference omitted raw fields. Changing the threshold does not disable the hard column budget.

SearchMemoryCompressor derives useful/rejected concepts, best features, successful/unsuccessful joint interactions, failed operators, invalid-proposal stage counts and recent frontier from inner trial evidence only. “Strong” is a positive joint parent delta, not a significance or causality claim. Entries remain bounded after 1000 trials. Invalid responses remain redacted; their arbitrary expressions are never stored as failed patterns.

`token_allocation` defaults to schema 0.40, memory 0.25, statistics 0.20, instructions 0.15. These are soft allocations; mandatory task/DSL/parent constraints may borrow unused capacity. The final prompt has a hard UTF-8-byte token reserve and character cap, including all repair suffixes. This intentionally conservative reserve is not billed token usage. Trimming removes detailed statistics, low-priority memory, lineage details, then lower-priority columns. Mandatory instructions are never silently truncated; an impossible budget fails before the API call.

`include_statistics=True` computes detailed target-free statistics lazily for retrieved fields, up to `max_detailed_stats_columns`, and reuses them within the optimize call. The builder does not accept outer-test data or scores. The caller must supply training rows only to optimize; an arbitrary DataFrame cannot be recognized as a test set automatically. Statistics omit category labels and raw rows.

Trial `context_metadata` persists selected names, scores/reasons, retrieval/build latency, prompt hash and token reserve, without the prompt or row data. Context/retriever/memory versions and configuration enter ExperimentFingerprint. Changing context settings rejects resume unless an explicit archive-and-restart is requested. `context_budget={...}` and `retrieval_threshold` can also be supplied under a CLI LLM sampler configuration.

## Correctness and cache compatibility (2026-09-22)

- Numeric inputs are promoted to floating point **before** arithmetic, including unsigned subtraction and integer squares. This avoids integer wrapping; non-finite floating outputs still become missing.
- Reusing CVEvaluator with changed X/y now raises before reading caches. Explicitly call `prepare(X, y, groups)` for a new dataset; FeatureStudy does so automatically. In-place data mutation is detected too.
- `content_hash` retains names for lineage, while `FeatureSet.expression_hash` recursively ignores display aliases for comparison. Trial deduplication, evaluation caches, fold matrices and fitted pipelines use name-sensitive identity because DataFrame estimators may inspect column names.
- Compiler version 3, prompt version 4 and context version 2 invalidate previous experiment identities. Old studies remain readable, but numerical continuation requires compatible fingerprints or explicit archive-and-restart.
- Malformed OpenAI-compatible responses, including empty choices, produce a recorded LLM failure instead of aborting the entire search.

## Default TabPFN model

Built with PriorLabs-TabPFN

`pip install featune` includes TabPFN, PyTorch and Plotly. The `visualization` and `torch` extras remain empty compatibility aliases. Use `pip install -e '.[dev]'` for development.

The default pins **tabpfn==9.0.0 and TabPFN-2**. Both task-specific revisions and SHA-256 hashes are recorded in `featune/tabpfn.py`. Choose `model_version="v3.5"` explicitly for the standard 3.5 checkpoint (`tabpfn-v3.5-20260909.safetensors`, revision `06bf2ba35c80a92a3b9abb436b99cf49e7a0365e`), shared by classification and regression. The package version is independent of the model generation. v2 licensing is in `LICENSES/TabPFN-v2.txt`; optional 3.5 licensing is in `LICENSES/TabPFN-3.5.txt`. v2 classifier weights are upstream-finetuned, so benchmark interpretation should consider possible training-data overlap.

Omitting estimator/evaluator in `optimize`, or using `CVEvaluator()`, chooses classification for auc/auc_ovr/accuracy/f1/log_loss and regression for rmse/mae/r2. Integer targets are not used to guess the task. Search direction is inferred from the metric; a contradictory explicit direction is rejected. Explicit estimators/evaluators always retain precedence.

```python
study = featune.create_study(metric="rmse", direction="minimize")
study.optimize(X, y, schema, n_trials=3)

evaluator = featune.CVEvaluator(
    featune.TabPFNRegressor(device="cpu", n_estimators=2, random_state=42),
    metric="rmse", cv=3,
)
```

Both wrappers accept `model_version="v2", n_estimators=4, device="auto", random_state=42`, support sklearn cloning/parameter APIs, and expose immutable model provenance. Ensemble size and seed do not follow upstream automatic defaults. Joint search can vary approved settings such as `n_estimators`; learning rate and epochs do not apply to this fixed-weight inference preset.

Fold-compiled DataFrames go directly to TabPFN without generic imputation, scaling or one-hot encoding. Each fold has independent training context. Default v2 capacity is 10,000 training rows, 500 columns and 2–10 classes; CPU supports 1,000 rows. Opt-in v3.5 raises row/column/CPU limits to 1,000,000/20,000/5,000; the wrapper still supports 2–10 classes. These are guardrails, not hardware capacity guarantees. Full-data refit is subject to the same limits. Exceeding limits raises an error: there is no silent subsampling, limit override, estimator fallback or cloud inference. Pass another estimator explicitly for larger datasets.

`importance_repeats=None` selects zero for the TabPFN adapter and two for other estimators. Explicit positive repeats enable explanations and require `n_jobs=1` to avoid copying device state. With explanations disabled, permutation importance is empty and evidence-dependent hypotheses may remain inconclusive. Trials record `prediction_time` and `explanation_time` in seconds; fit counts are API calls, not gradient steps or total compute.

Default v2 first fit downloads approximately 29 MB (classifier) or 44 MB (regressor), without the 3.5 browser-authorization flow. For opt-in v3.5 only, first uncached fit calls the official license flow before downloading approximately 876 MB. In an interactive terminal with a desktop browser, it opens the login page automatically and waits for the browser callback; after the user logs in and accepts the license, download resumes automatically. Featune never accepts terms for the user. In notebooks/non-interactive/headless sessions, complete authorization at https://ux.priorlabs.ai first and configure `TABPFN_TOKEN`; use `TABPFN_NO_BROWSER=1` to fail promptly when credentials are missing. Configure `HF_TOKEN` separately if Hugging Face requests repository access. Never commit tokens. Installation does not download weights or grant model rights.

Checkpoint resolution does not upload training rows. After an authorized small fit, set `HF_HUB_OFFLINE=1` to test cached operation; a cache miss fails without browser/authentication requests. `HF_HOME` controls cache location. Only v3.5 classification and regression share a weight file. Network access for an explicitly configured LLM sampler is separate from local TabPFN inference.

Joblib exports/cache entries contain an upstream fitted-state archive, excluding foundation weights. Restoration resolves and verifies the fixed weights on the destination machine and rewrites the original machine's cache path. The destination needs the same package version and cached weights, or download access on first prediction. `device="auto"` adapts to available hardware; an explicit GPU device remains explicit. Before first prediction, use `pipeline.set_params(model__device="cpu")` to restore a GPU-configured artifact on CPU. Cross-device numerical differences are possible.

**Fitted TabPFN artifacts contain training features and labels as context.** Protect model/cache files as training data, not anonymized parameters, and load only trusted joblib artifacts. This does not add raw rows to HTML reports or SQLite search history. Follow the bundled weight license when distributing downstream models or services.


### License and operational constraints

Featune code remains MIT; TabPFN package code is separately Apache-2.0. The following restrictions apply to **optional v3.5**. Default v2 uses its separate Apache-2.0-based Prior Labs license with attribution and downstream model-naming requirements; see the bundled text. Model rights are granted directly by Prior Labs under the [TabPFN-3.5 License v1.0](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/06bf2ba35c80a92a3b9abb436b99cf49e7a0365e/LICENSE), revised September 9, 2026. This summary does not replace those terms.

- Qualifying non-commercial research, testing and limited internal evaluation are allowed. Commercial or production use requires a separate commercial license, even when Featune or your application is open source.
- Hosted/managed services, APIs and SaaS require separate commercial permission, whether paid or free. Outputs used in production, business processes or client deliverables are restricted too; outputs cannot train/fine-tune/distill a competing model.
- The license excludes consumers as defined by German BGB §13 and requires professional status under §14. Redistribution has license-copy, attribution and derivative-notice conditions; include the upstream license and `NOTICE` where applicable. Contact Prior Labs for applicable permissions.
- These are preset guardrails, not a promise that maximum-sized data fits available hardware. The wrapper deliberately supports 2–10 classes, and CV/refit must each fit memory. Fixed-weight inference has no learning-rate/epoch training controls.
- Switching between v2 and v3.5 changes study provenance. Use a new study directory and refit; keep the original environment for artifacts created before the model-selection API. Never edit stored provenance to force reuse.
- Real-weight acceptance tests require authorized model access. Ordinary tests use controlled doubles and do not establish numerical quality or real-weight export compatibility. Run the opt-in integration suite before release.


### Explicit model selection

```python
model = featune.TabPFNClassifier(model_version="v3.5")
study.optimize(X, y, schema, estimator=model, n_trials=3)
```

CLI JSON:

```json
{"estimator": {"type": "tabpfn", "params": {"model_version": "v3.5"}}}
```

Omit `model_version` for v2. Regression uses `TabPFNRegressor` with the same option.
