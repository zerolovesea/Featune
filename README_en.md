# Featune

Built with PriorLabs-TabPFN

**Auditable feature-engineering search for tabular data.**

Featune puts field semantics, candidate features, cross-validation, and resource budgets into one resumable search loop. It runs without an LLM through random/evolutionary search, or uses an LLM for business hypotheses. Every proposal passes a constrained DSL, compiler, and independent evaluator; model-generated code is never executed directly.

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" /></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB.svg?logo=python&logoColor=white" alt="Python 3.10+" />
</p>

[中文](README.md) · English · [Guide](docs/en/guide.md) · [DSL/API](docs/en/reference.md) · [Results](benchmarks/README.md) · [Kaggle examples](examples/kaggle/README.md)

## Why Featune

Classical AutoFE is good at enumerating numeric transformations. LLMs are good at proposing domain hypotheses. Both can leave an opaque search trail. Featune treats every idea as a controlled, measurable, resumable experiment:

| Problem | Featune approach |
|---|---|
| The LLM writes arbitrary code | JSON proposals are compiled through a whitelisted FeatureIR |
| A plausible feature does not improve the model | The baseline remains eligible and candidates must pass inner CV |
| Leakage enters feature construction | Learned transforms fit inside training folds; the outer test is evaluated once |
| A failed trial cannot be explained | State, parent feature set, hypothesis, error, and resource usage are persisted |
| Runs cannot be compared | Dataset/config/dependency fingerprints and split hashes are recorded |
| LLM context or cost grows without control | Token/cost/time budgets, bounded history, caching, and explicit failure states |

Featune is not a wrapper that promises to beat OpenFE. Its differentiator is **constraints, semantics, evidence, and recovery**. Accuracy claims still require outer holdouts and multiple fixed seeds.

### Relationship to brute-force interactions

When a dataset has few numeric columns and the only goal is score, enumerating second-order interactions or using a GBDT is often faster; Featune is not meant to replace that baseline. Featune targets what brute-force combinations handle poorly: domain semantics, categorical/temporal/group constraints, leakage boundaries, feature hypotheses, failure records, budgets, and resume. Its goal is a **controlled, explainable, reproducible experiment**, not a promise of higher AUC on every dataset.

## Architecture

~~~mermaid
flowchart LR
  A[Dataset + DatasetSchema] --> B[Study]
  B --> C{Sampler}
  C -->|Random / Evolution| D[Feature proposal]
  C -->|LLM / Hybrid| E[JSON proposal + hypothesis]
  D --> F[FeatureIR / FeatureSet]
  E --> F
  F --> G[DSL validation + compiler]
  G --> H[Fold-local pipeline]
  H --> I[Inner CV evaluator]
  I --> J[Trial state + metrics + lineage]
  J --> B
  J --> K[SQLite / cache / fingerprint]
  B --> L[Exported sklearn pipeline]
  J --> M[HTML report / Pareto / efficiency]
~~~

Boundaries:

- **Schema** describes field names, types, units, domain meaning, objectives, and unavailable fields. Raw rows and labels never enter LLM prompts.
- **Samplers** propose candidates; they do not decide validity. Random, evolution, LLM, and hybrid samplers share one verification path.
- **IR/compiler** turns proposals into sklearn pipelines with operator, type, dependency, and fold-local checks.
- **Evaluator** owns stratified, grouped, temporal, or custom CV.
- **Storage** persists trials, FeatureSet hashes, lineage, cache entries, budgets, and fingerprints for safe resume.
- **Reporting** exposes resource efficiency, hypotheses, feature evolution, correlations, and Pareto frontiers.

## When to use it

Use Featune when:

- columns have domain meaning and search should use that context;
- you need to audit why a feature was proposed, whether it helped, and what it cost;
- data requires categorical, grouped, or temporal validation constraints;
- a search may run for hours and must resume, deduplicate, budget, and record failures;
- one experiment needs offline baselines, LLM semantics, and external comparators.

A plain sklearn Pipeline is better for one or two known transforms. Featune does not replace data governance, causal analysis, or a feature store, and it should not enumerate unbounded candidates over billions of rows without sampling or aggregation.

## Install

Python 3.10+:

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install featune
~~~

For examples and benchmarks run from a repository checkout, install optional extras:

~~~bash
pip install -e '.[benchmark]'                 # OpenFE, LightGBM, public-data benchmarks
pip install -e '.[dev]'   # development and full tests
~~~

LightGBM on macOS may require brew install libomp. Local examples require no LLM API key; model weights must be cached before offline use.

The default installation includes TabPFN, PyTorch and Plotly. The default preset pins **tabpfn==9.0.0 and TabPFN-2**, with immutable classifier/regressor revisions and SHA-256 hashes. Limits: 10,000 training rows, 500 columns, 2–10 classes; CPU allows 1,000 rows. First fit downloads public v2 weights without v3.5 account authentication, subject to the v2 license.

Select `featune.TabPFNClassifier(model_version="v3.5")` (or the regressor) to opt into the fixed 3.5 preset. In an interactive terminal, first download opens the official login page; after the user accepts the terms and the browser callback returns, downloading resumes automatically. Non-interactive/headless environments require preconfigured credentials. The 3.x option specifically means 3.5, not 3.0.

**Featune's MIT license does not cover model weights.** v2 has its own license and attribution requirements. Optional 3.5 is restricted to qualifying non-commercial, non-production use; commercial/production use and hosted/API/SaaS services (paid or free) need separate authorization, with restrictions on outputs too. See [model selection and license details](docs/en/reference.md#default-tabpfn-model).

## Quickstart

~~~python
import numpy as np
import pandas as pd
import featune

rng = np.random.default_rng(42)
X = pd.DataFrame({"loan": rng.uniform(1, 20, 200), "income": rng.uniform(1, 10, 200)})
y = (X["loan"] / X["income"] > 2).astype(int)

schema = featune.DatasetSchema(fields=[
    featune.FieldSchema(name="loan", description="Loan balance at application", unit="CNY"),
    featune.FieldSchema(name="income", description="Monthly disposable income", unit="CNY/month"),
], objective="Predict high debt burden")

study = featune.create_study(metric="auc", sampler=featune.RandomSampler(seed=42))
study.optimize(X, y, schema=schema, n_trials=3)
print(study.best_value)
print(study.best_features)
pipeline = study.export_pipeline()
~~~

best_value is the inner CV score. Keep an untouched outer test set for final evaluation:

~~~bash
python examples/quickstart.py
featune optimize examples/config.json
featune report --storage runs --study-name cli-demo --output runs/cli-demo/report.html
~~~

The [Chinese notebook](examples/quickstart_zh.ipynb) and [English notebook](examples/quickstart_en.ipynb) explain data splitting, Schema, search, evaluation, evidence, export and an optional LLM run. Start Jupyter from the repository root or `examples/`; the LLM cell requires `RUN_LLM=True` and configured environment variables.

### Kaggle Playground with default TabPFN v2

Both Playground notebooks use Featune's default TabPFN v2. Its single-fit limit is 10,000 rows with an accelerator or 1,000 on CPU, so each notebook reserves an independent holdout, samples the training rows with stratification, and predicts **every test row** in batches. Each run prints its actual training sample size; it does not claim full-data training. Featune compares raw fields and controlled features on the same cross-validation folds. The [Kaggle examples](examples/kaggle/README.md) cover data loading, Schema, search, validation, submissions, and reproducible runs.

Public notebooks: [S6E9 EV purchase prediction](https://www.kaggle.com/code/yaaangzhou/featune-ev-purchase-with-tabpfn-v2) · [S6E8 smartphone addiction prediction](https://www.kaggle.com/code/yaaangzhou/featune-smartphone-addiction-tabpfn-v2). Run records and submission scores are in the examples guide.

## LLM semantic search

~~~bash
export FEATUNE_API_KEY='your-key'
export FEATUNE_MODEL='your-model-id'
python examples/claude_search.py --trials 3
~~~

~~~python
sampler = featune.LLMSampler(
    provider="anthropic", max_tokens=2048,
    token_budget=100_000, features_per_trial=1,
)
study = featune.create_study(sampler=sampler, storage="runs", study_name="credit")
study.optimize(X, y, schema=schema, n_trials=3)
~~~

Prompts contain schema, objectives, controlled feature definitions, trial metrics, and hypotheses—not raw rows or labels. Responses must be valid JSON and pass duplicate-key/non-finite checks, Pydantic validation, FeatureIR whitelisting, and actual CV before entering the best feature set. Anthropic-compatible endpoints append /v1/messages; OpenAI-compatible and local endpoints are documented in the [API reference](docs/en/reference.md).

## Capabilities

| Layer | Capability |
|---|---|
| Schema | Field semantics, types, units, objectives, partitions, excluded fields |
| FeatureIR | Stable identity, dependencies, hashes, parent/child lineage |
| Compiler | 18 controlled operators, type checks, fold-local statistics, sklearn pipelines |
| Search | Baseline, random, evolution, LLM, hybrid, greedy/beam, deduplication, early stopping |
| Evaluation | Classification/regression, multiple metrics, stratified/group/time/custom CV |
| Reliability | SQLite transactions, single-writer lock, six cache layers, budgets, resume, fingerprints |
| Evidence | Trial states, hypotheses, failures, feature evolution, resource tables, Pareto frontiers |
| PyTorch | MLPs, categorical embeddings, joint architecture/feature search |
| Interfaces | Python, Jupyter, CLI, fitted pipeline export, offline HTML reports |

## Metrics and protocol

Featune stores more than the final score:

| Metric | Meaning |
|---|---|
| test_score | Final outer-holdout score; evaluated once for the selected pipeline |
| inner_best / inner_baseline | Best and original score inside the search CV |
| evaluations_to_best | Candidate evaluations needed to reach the best result |
| fixed_gain_* | Evaluations/time/cost needed to hit a fixed absolute improvement |
| 95pct_final_gain_* | Resources needed to reach 95% of the final inner gain |
| tokens / api_cost | LLM usage and cost when explicit prices are supplied |
| valid_proposal_rate | Proposals that pass protocol validation and enter evaluation |
| model_fits | CV and candidate fits; external frameworks are marked unknown when internal counts are hidden |

Protocol rules:

- Binary classification uses outer ROC AUC; multiclass uses weighted one-vs-rest AUC.
- Regression uses outer RMSE.
- Methods share the outer split hash; search only sees inner CV on the training portion.
- A stable improvement requires multiple predeclared seeds, not the best single run.
- A semantic improvement requires a correct-vs-shuffled/blinded comparison, not only baseline comparison.
- An efficiency claim reports time, fits, tokens, and cost together.

## Public benchmark evidence

### CoverType: OpenFE vs Featune

581,012 rows, 54 fields, 7 classes; fixed linear downstream model, seed 22, two Featune trials, and 32 OpenFE row-local numeric candidates:

| Method | Weighted OVR AUC | Seconds |
|---|---:|---:|
| Featune baseline | 0.871745 | 24.15 |
| Featune random | 0.871727 | 65.53 |
| Featune evolution | 0.871727 | 58.79 |
| OpenFE | **0.873956** | 83.19 |

OpenFE scores higher in this constrained numeric comparison. This is not evidence that it dominates Featune across its full search space, multiple seeds, or semantic tasks.

### TabPFN v2: six datasets, five seeds

All 240 runs completed: raw v2, Random, Evolution, semantic/anonymous LLM, restricted OpenFE, and raw-feature linear/gradient-boosting controls. Each dataset uses at most 1,200 rows, a 25% outer holdout, 3-fold inner CV and up to four feature proposals; these are not full-data scaling results.

Semantic LLM search improved Adult AUC by 0.229 percentage points on average and reduced California RMSE by a paired average of 0.927%, but its descriptive improvement intervals include zero on all six tasks. Adult Random improved all five seeds, averaging +0.197 AUC percentage points. Traceable expressions and validation evidence are demonstrated capabilities; LLM domain rationales still need human review. See the [complete v2 report](benchmarks/results/tabpfn-v2/report.md) for scores, costs and limitations.

### Online Retail II: historical results withdrawn

The previous two/five-seed runs clipped targets at a full-data 99.5th percentile before splitting, leaking held-out target information. Their gain and semantic-effectiveness conclusions are withdrawn. Raw files have been removed; the withdrawal rationale remains in the historical summary. Invalidated scores must not be compared with corrected runs.

The loader now preserves original purchased quantities without data-dependent clipping. The v2 report above includes the corrected five-seed small-sample semantic ablation; withdrawn full-data results remain invalid for benefit claims. See [benchmark documentation](benchmarks/README.md) for earlier smoke and wide-context acceptance.

## Relationship to other frameworks

| Framework | Strength | Featune boundary |
|---|---|---|
| [OpenFE](https://github.com/IIIS-Li-Group/OpenFE) | Large row-local numeric candidate screening | Featune does not reproduce every operator; comparisons cap candidates and report hidden internal counts |
| [CAAFE](https://github.com/noahho/CAAFE) / [paper](https://arxiv.org/abs/2305.03403) | LLM-generated feature code with iterative verification | Featune never executes generated code and uses constrained IR; CAAFE focuses on small classification data, not million-row workloads |
| [Featuretools/DFS](https://github.com/alteryx/featuretools) | Multi-table relational aggregation | Featune currently focuses on controlled single-table search; relational adapters are a separate extension |
| AutoFeat and formula search | Automatic numeric formulas | Featune emphasizes schema, budgets, lineage, CV contracts, and failure auditability |

Choose by task rather than brand ranking: evaluate OpenFE for large row-local numeric enumeration, DFS for relational tables, and Featune when semantic context, constraints, recovery, and auditability matter. Reproduce comparisons with the same holdout and downstream estimator.

## Limitations

- Search is not guaranteed to improve; the baseline can win.
- Current LLM experiments do not establish general accuracy or execution-efficiency superiority.
- The OpenFE comparator is row-local and constrained; older releases may be incompatible with current NumPy/scikit-learn versions.
- Hypotheses and associations are not causal claims or SHAP values.
- Users must exclude unavailable future information and choose appropriate CV for grouped or temporal data.
- Remote services can have unknown billing and model drift; budgeted runs stop when the next request cannot be safely reserved.

## Documentation and development

- [Guide](docs/en/guide.md) · [API reference](docs/en/reference.md)
- [Full Kaggle Playground examples](examples/kaggle/README.md)
- [Benchmark methods/results](benchmarks/README.md)
- [Layered comparison plan](benchmarks/COMPARISON_PLAN.md)
- [Contributing](CONTRIBUTING.md)

~~~bash
pytest -q
ruff check featune tests examples benchmarks
python -m build
~~~

MIT License.
