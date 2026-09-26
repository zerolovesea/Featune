# User guide

[中文](../zh/guide.md) · [Home](../../README_en.md) · [API](reference.md)

## Data and semantics

New users can start with the [quickstart](../../README_en.md#quickstart). The [Playground notebooks](../../examples/kaggle/README.md) show full-dataset training and competition submissions. These datasets exceed the default TabPFN v2 row limit, so the notebooks explicitly pass LightGBM and refit the selected pipeline on every training row after feature search.

`X` is a pandas DataFrame with unique column names; `y` is an equally sized array or a Series with matching indexes. Keep targets out of usable fields. Schemas accept model objects or matching dictionaries:

```python
from featune import DatasetSchema, FieldSchema
schema = DatasetSchema(
    fields=[
        FieldSchema(name="loan", description="Principal at application", unit="CNY", partition="finance"),
        FieldSchema(name="income", description="Monthly income at application", unit="CNY/month", partition="finance"),
        FieldSchema(name="region", description="Applicant region", dtype="categorical", partition="profile",
                    encoding="Administrative region code; missing means not reported",
                    available_at="Application submission"),
        FieldSchema(name="settled_at", description="Future settlement time", dtype="datetime", exclude=True),
    ],
    partitions={"finance": "Financial information before the lending decision", "profile": "Applicant profile"},
    objective="Predict future delinquency using information available at application",
    target_definition="1 means at least one delinquency within 90 days after application; 0 means none",
    prediction_point="At application submission, before disbursement or repayment",
)
```

Excluded fields cannot enter the model or be referenced by expressions. Columns absent from the schema are ignored. Partitions describe business domains, not train/test subsets. The objective, target definition, prediction point, field encoding and availability are sent to LLM prompts without row-level labels or samples. They do not automatically detect future leakage; exclude post-prediction fields explicitly. Prompt descriptions retain only their first 240 characters, so put important code meanings in `encoding`.

After feature computation, nonfinite numbers become missing and fold-local median imputation handles numeric columns, preserving empty columns. Categories use type-tagged string keys that distinguish missing values and mixed Python types. The sklearn path uses one-hot encoding with at most 128 categories per field and ignores unseen categories. Dates become UTC; original date fields become days since the Unix epoch. Type mismatches fail before training.

## Samplers

```python
import featune
random_sampler = featune.RandomSampler(seed=42, features_per_trial=2)
evolution_sampler = featune.EvolutionSampler(seed=42, features_per_trial=2)
llm_sampler = featune.LLMSampler(provider="anthropic", model="claude-opus-4-8", features_per_trial=2)
hybrid_sampler = featune.HybridSampler(llm_sampler=llm_sampler, random_sampler=evolution_sampler, llm_every=3)
```

Random uses trial-indexed RNG for restart reproducibility and can construct higher-order features from the parent. Evolution adds elite-expression recombination to random mutation. LLM consumes field/partition semantics, objectives, parent features, scores, importances and failed trials; the default memory retains up to eight recent records plus bounded best-feature/concept summaries before token and character trimming. Hybrid calls the LLM at trials 1, 4, 7… and its other sampler between them.

Custom samplers implement `BaseFeatureSampler.sample(context) -> Proposal`. Stateful implementations should override `state_dict/load_state_dict`; include all proposal-affecting configuration in `configuration()` so resume validation remains meaningful.

## Independent evaluation

```python
from featune import CVEvaluator

evaluator = CVEvaluator(
    metric="auc", cv=5,
    metrics=["accuracy", "log_loss"], random_state=42,
    importance_repeats=2, importance_max_samples=500, n_jobs=1,
)
```

Integer CV uses StratifiedKFold for classifiers and KFold for regressors, or GroupKFold when groups are supplied. TimeSeriesSplit, sklearn splitters and explicit `(train_indexes, validation_indexes)` lists are supported. Sort time series first and set an appropriate gap for label availability. Group overlap is rejected.

```python
from sklearn.model_selection import TimeSeriesSplit
time_evaluator = CVEvaluator(cv=TimeSeriesSplit(n_splits=4, gap=7))
```

Each fold clones the model and fits feature statistics, imputation, encoding and scaling only on training rows. All trials use the same folds. Permutation importance shuffles compiled feature columns on validation rows, using the higher-is-better scorer convention. Original and derived features are shuffled separately, so it is not a total effect with descendants recomputed.

For tables beyond the default TabPFN capacity, supply a sklearn-compatible estimator explicitly. Here `X_search/y_search` excludes an independent holdout. After checking that holdout using the exported fitted pipeline, clone the selected structure and refit it on every official training row:

```python
from lightgbm import LGBMClassifier
from sklearn.base import clone

evaluator = featune.CVEvaluator(
    estimator=LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1),
    metric="auc", cv=2, importance_repeats=0,
)
study = featune.create_study(metric="auc", sampler=featune.RandomSampler(seed=42))
study.optimize(X_search, y_search, schema=schema, evaluator=evaluator, n_trials=2)
search_pipeline = study.export_pipeline()  # already fitted on X_search
final_pipeline = clone(search_pipeline).fit(X_all, y_all)  # all labeled rows
```

`importance_repeats=0` saves time but produces no permutation-importance evidence. LightGBM is an optional dependency of this example, not part of Featune's default install. See the [Kaggle examples](../../examples/kaggle/README.md) for loading, validation and submission.

## Search and budgets

```python
study = featune.create_study(
    metric="auc", direction="maximize", sampler=random_sampler,
    search_strategy="beam", beam_width=3, max_features=16,
    storage="runs", study_name="credit-v1", joint_optimization=True, param_space={"C": [0.1, 1.0, 10.0]},
)
# X/y/schema must describe the same training dataset.
study.optimize(X, y, schema=schema, evaluator=evaluator, n_trials=30, timeout=3600, patience=10)
```

Independent search expands the baseline. Greedy expands the best expandable set. Beam freezes a Top-K frontier for each generation and expands its members in turn. The baseline remains eligible. Multiple proposed features may depend on earlier entries in the same list.

`n_trials` means additional attempts in this call, excluding baseline. Failed and duplicate attempts count toward the budget. `max_features` counts generated features only. Patience counts consecutive non-improving attempts within this call. Timeout includes baseline time but is checked at trial boundaries; it does not interrupt an estimator already training, and final refit is outside the trial scheduling deadline.

Model parameters must come from finite user-approved `param_space` choices. A trial can change model settings and features together, so its delta describes that combined change rather than an isolated feature effect.

```python
def on_trial(study, trial):
    print(trial.number, trial.state, trial.value, trial.tokens)
    if trial.value is not None and trial.value >= 0.99:
        study.stop()
```

Pass `callbacks=[on_trial]`. Callbacks run after persistence; call `stop()` to request a stop. KeyboardInterrupt is not swallowed, and completed trials are already committed.

## Resume

```python
sampler = featune.RandomSampler(seed=42, features_per_trial=2)
study = featune.load_study("runs", "credit-v1", sampler=sampler)
study.optimize(X, y, schema=schema, evaluator=evaluator, n_trials=20)
```

Use the original data, schema, estimator, folds, importance settings and sampler configuration. Keys may change and budgets may increase. Loading without a sampler is useful for read-only inspection; recreate the original sampler to continue non-default searches.

`runs/<study_name>/` contains SQLite metadata/history, `search.log` and an OS-managed writer lock. The log and CLI's default INFO output show each completed fold, every validated proposed hypothesis (concept, rationale, operation, inputs and expected direction), its assessed status and reason, and the trial's best-selection decision. Invalid raw LLM responses are not logged; only sanitized errors are recorded. Library callers can configure `logging.getLogger("featune")`. Logs may contain business field semantics and LLM rationale. The CLI also writes `pipeline.joblib`, `features.json`, `trials.json` and optionally `report.html`. Raw X/y are not saved in history. Fitted model artifacts do contain learned parameters and statistics and should be handled as training-data derivatives.

Abandoned RUNNING trials become INTERRUPTED and keep their numbers. Requests that may already have been billed are not replayed. Unknown billing is recorded; an active token budget blocks additional LLM calls after uncertainty. Reconcile with the provider before starting a new budgeted study. Unknown consumption is not zero consumption.

## LLM protocols and accounting

LLMClient supports Anthropic Messages, OpenAI Chat Completions and local OpenAI-compatible servers. Only semantic definitions and summarized trial feedback are sent, never raw rows or labels. Temperature is omitted by default; a service may reject an explicitly supplied value. OpenAI uses `max_completion_tokens`; legacy gateways may set `output_token_parameter="max_tokens"`.

```python
openai_sampler = featune.LLMSampler(provider="openai", model="your-chat-model")
gateway_sampler = featune.LLMSampler(provider="openai", model="your-model",
                                    base_url="https://gateway.example/v1",
                                    output_token_parameter="max_tokens")
claude_sampler = featune.LLMSampler(provider="anthropic", model="your-claude-model",
                                   base_url="https://gateway.example")
```

These examples read credentials from `FEATUNE_API_KEY`. Custom services must return the matching protocol's response and usage fields. OpenAI Responses API and gateway-specific payload fields are outside this compatibility interface.

Input/output usage includes JSON repair attempts. By default, one repair is allowed after the initial response. Unknown fields/operators, invalid references or types, duplicate JSON keys, nonfinite values and unapproved model parameters are rejected. Only JSON or a single `json` fenced block is accepted; arbitrary prose is not searched for embedded JSON.

Requests have a 60-second default timeout. Rate limits and selected server failures have bounded retries. Transport failures are not automatically retried because billing may be unknown. HTTP errors log status only; response bodies, credentials and invalid model output are not persisted. The token budget reserves UTF-8 prompt bytes plus max_tokens before a call; this is conservative scheduling, not exact tokenization or a provider-enforced billing cap. Actual usage comes from the provider, and missing usage remains unknown.

`client.cost` estimates cost using user-supplied input/output prices per million tokens. It does not fetch current pricing or apply special cache discounts. Cache tokens count toward input totals. Network failures or process termination can leave actual billing higher than reported known usage.

Remote endpoints require HTTPS, local HTTP is limited to loopback, and redirects are disabled. Set FEATUNE_API_KEY instead of embedding credentials in CLI JSON. Field descriptions can themselves be sensitive; choose their content and provider accordingly.

## PyTorch

```python
from featune.torch import TorchClassifier
network = TorchClassifier(hidden_dims=(64, 32), embedding_dim=8, learning_rate=0.001,
                          epochs=30, batch_size=128, device="cpu", random_state=42)
network_study = featune.create_study(
    sampler=featune.RandomSampler(seed=42),
    joint_optimization=True, param_space={"hidden_dims": [[32], [64, 32]], "learning_rate": [0.001, 0.003],
                 "embedding_dim": [4, 8]},
)
network_study.optimize(X, y, schema=schema, estimator=network, n_trials=10)
```

The PyTorch path receives compiled DataFrames directly. Fold-local category dictionaries feed trainable embeddings, with index zero reserved for unseen values. Numeric and regression-target scaling are learned only on training rows. Use TorchRegressor for regression. Hidden layers, learning rate, embedding width, dropout, weight decay, epochs and batch size are tunable. CPU is covered by automated checks; GPU determinism depends on hardware/PyTorch settings. Other sklearn-compatible neural estimators can be passed directly.

## Evidence and export

```python
study.best_trial
study.best_features
study.best_value
study.trials_dataframe()
study.feature_history()
study.concept_summary()
study.report("runs/report.html")
study.export_pipeline("runs/model.joblib")
study.export_features("runs/features.json")
```

Reports embed Plotly and work offline. Display them in notebooks with `IPython.display.HTML(study.report())`. Concept counts are trial records, including repeated features, not independent scientific replications.

`export_pipeline()` returns the best pipeline refitted on all rows supplied to the most recent optimize call. It applies identical transformations during inference. `refit=False` does not create an exportable fitted model. A read-only study load cannot reconstruct training data; use the original configuration and `optimize(..., n_trials=0)` to refit. Feature JSON contains reviewable DSL, not model weights.

Load joblib artifacts only from trusted sources: Python pickle can execute code. Match training dependency versions in the inference environment.

## Wide-schema search

Use `semantic_tags`, field descriptions and the task objective to make local retrieval useful. `LLMSampler(context_budget={"max_columns": 40, "max_tokens": 12000})` automatically retrieves a bounded subset. Add `include_statistics=True` to the study for selective training-only summaries. Inspect `study.trials[-1].context_metadata` for selection reasons and prompt-budget diagnostics. See the [context API and limitations](reference.md#wide-schemas-and-bounded-context).

Offline retrieval acceptance: `python benchmarks/wide.py --output runs/wide-smoke`. This does not spend API tokens or establish predictive gains. Real LLM evaluation requires the explicit `--llm` flag.


Built with PriorLabs-TabPFN

The default is pinned TabPFN-2. See the [reference](reference.md#default-tabpfn-model) for installation, capacity, offline weights and artifact handling. TabPFN artifacts contain training examples and labels as context.
