# Featune

Auditable feature search for tabular data. Featune proposes features through a constrained DSL, evaluates them with cross-validation, records each trial, and exports a fitted scikit-learn pipeline. Random and evolutionary search work without an LLM; Anthropic Messages and OpenAI Chat Completions are optional proposal sources.

## Install

```bash
pip install featune
```

Python 3.10 or newer is required. The default model uses pinned TabPFN-2 weights downloaded on first use; the weights are not included in this package. Review `NOTICE` and the model licenses in `LICENSES/` before using TabPFN. TabPFN-3.5 is an opt-in model with separate access and use restrictions.

## Minimal example

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import featune

data, labels = make_classification(
    n_samples=120, n_features=2, n_informative=2, n_redundant=0, random_state=42
)
X = pd.DataFrame(data, columns=["signal_a", "signal_b"])
y = pd.Series(labels)
schema = featune.DatasetSchema(
    fields=[
        featune.FieldSchema(name="signal_a", description="First synthetic signal"),
        featune.FieldSchema(name="signal_b", description="Second synthetic signal"),
    ],
    objective="Predict a synthetic binary label",
)
train, test = train_test_split(range(len(X)), stratify=y, random_state=42)
study = featune.create_study(metric="auc", sampler=featune.RandomSampler(seed=42))
study.optimize(X.iloc[train], y.iloc[train], schema, evaluator=featune.CVEvaluator(cv=2), n_trials=1)
print(study.trials_dataframe())
```

Use a larger dataset and reserve an untouched test set for real evaluation. See the [English notebook](https://github.com/zerolovesea/Featune/blob/main/examples/quickstart_en.ipynb), [Chinese notebook](https://github.com/zerolovesea/Featune/blob/main/examples/quickstart_zh.ipynb), and [API guide](https://github.com/zerolovesea/Featune/blob/main/docs/en/guide.md). LLM credentials are read from `FEATUNE_API_KEY`; keep them outside source files and notebooks.

For LLM search, add `search_guidance="Try missing-value flags and useful ratios"` to `DatasetSchema` and use `LLMSampler`. Guidance affects column selection and proposals; the closed DSL and cross-validation still validate each candidate. Guidance is sent to the configured LLM provider, so omit secrets and private data.

Featune source code is MIT licensed. TabPFN code and model weights have separate licenses; see the included notices. Generated pipelines may retain training data and should be handled accordingly.
