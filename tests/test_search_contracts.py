# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Production IR, budgets, cache and accounting acceptance.

Exercises the module contracts with deterministic local fixtures. External LLM
requests are mocked; temporary storage keeps acceptance runs isolated.

Created:
    2026-09-22
"""

import json

import httpx
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from test_core import RatioSampler, dataset, feature

from featune import CVEvaluator, FeatureSet, LLMSampler, SearchBudget, create_study, load_study
from featune.cache import ArtifactCache
from featune.diagnostics import pareto_frontier, search_diagnostics
from featune.llm import BudgetExceeded, LLMClient


class NameModel(RegressorMixin, BaseEstimator):
    """Deliberately name-sensitive DataFrame estimator for cache regression checks."""

    accepts_dataframe = True

    def fit(self, X, y):
        """Mark this test estimator fitted."""
        self.fitted_ = True
        return self

    def predict(self, X):
        """Use a derived column only when it has the expected display name."""
        import numpy as np

        return X["good"].to_numpy() if "good" in X else np.zeros(len(X))


def evaluator():
    """Construct a cheap deterministic evaluator for recovery/cache acceptance.

    Returns:
        CVEvaluator: Two-fold logistic classifier with permutation explanation disabled.
    """
    return CVEvaluator(LogisticRegression(max_iter=1000), cv=2, importance_repeats=0)


def test_ir_identity_lineage_and_tampering():
    """Verify canonical graph identity, serialization and rejection of tampered/cyclic lineage.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    _, _, schema = dataset()
    first, second = feature(), feature("squared_ratio", "square", ["ratio"])
    forward = FeatureSet.from_features(schema, [first, second], source_trial=3)
    reverse = FeatureSet.from_features(schema, [second, first], source_trial=4)
    assert forward.content_hash == reverse.content_hash
    assert forward.features[1].generation == 2
    assert FeatureSet.model_validate_json(forward.model_dump_json()) == forward
    raw = forward.model_dump()
    raw["features"][1]["generation"] = 1
    with pytest.raises(ValueError, match="generation"):
        FeatureSet.model_validate(raw)
    raw = forward.model_dump()
    raw["features"][0]["params"] = {"fake": 1}
    with pytest.raises(ValueError):
        FeatureSet.model_validate(raw)
    with pytest.raises(ValueError, match="Cyclic"):
        FeatureSet.from_features(schema, [feature("a", "square", ["b"]), feature("b", "square", ["a"])])


def test_persistent_cache_and_resume_budgets(tmp_path):
    """Verify cumulative fit limits, disk cache reuse and explicit incompatible-study archiving.

    Args:
        tmp_path (pathlib.Path): Pytest-managed temporary directory isolating persisted artifacts.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    X, y, schema = dataset()
    study = create_study(
        sampler=RatioSampler(),
        search_strategy="independent",
        storage=tmp_path,
        budget=SearchBudget(max_trials=3, max_model_fits=5),
    )
    study.optimize(X, y, schema, evaluator=evaluator(), n_trials=2)
    assert study.consumption["model_fits"] == 5
    assert study.trials[-1].cache_hit
    cache_directory = study._cache.directory
    assert {path.name for path in cache_directory.iterdir()} == ArtifactCache.layers
    resumed = load_study(tmp_path, sampler=RatioSampler())
    assert resumed.fingerprint == study.fingerprint
    resumed.optimize(X, y, schema, evaluator=evaluator(), n_trials=3)
    assert resumed.consumption["model_fits"] == 5
    assert resumed.stop_reason == "max_model_fits"
    assert hasattr(resumed, "pipeline_")
    changed = X.copy()
    changed.loc[0, "loan"] += 1
    with pytest.raises(ValueError, match="Resume rejected"):
        resumed.optimize(changed, y, schema, evaluator=evaluator(), n_trials=0)
    resumed.optimize(changed, y, schema, evaluator=evaluator(), n_trials=0, allow_incompatible_resume=True)
    assert len(resumed.trials) == 1 and resumed.consumption["model_fits"] == 3
    with resumed.storage.connect() as db:
        assert db.execute("SELECT count(*) FROM archives").fetchone()[0] == 1


def test_evaluation_cache_avoids_all_fits(tmp_path):
    """Verify a saved evaluation avoids retraining while a changed cache namespace invalidates it.

    Args:
        tmp_path (pathlib.Path): Pytest-managed temporary directory isolating persisted artifacts.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    X, y, schema = dataset()
    ev = evaluator()
    ev.cache = ArtifactCache(tmp_path, "v1")
    ev.prepare(X, y)
    first = ev.evaluate(X, y, schema, [feature()])
    ev.cache = ArtifactCache(tmp_path, "v1")
    second = ev.evaluate(X, y, schema, [feature()])
    assert first.model_fits == 2 and second.model_fits == 0 and second.cache_hit
    assert first.fold_values == second.fold_values
    ev.cache = ArtifactCache(tmp_path, "v2")
    assert ev.evaluate(X, y, schema, [feature()]).model_fits == 2


def test_expression_aliases_keep_name_sensitive_evaluations(tmp_path):
    """Verify aliases keep separate scores for DataFrame estimators.

    Args:
        tmp_path (pathlib.Path): Pytest-managed temporary directory isolating persisted artifacts.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    from featune import Proposal
    from featune.samplers import BaseFeatureSampler

    class AliasSampler(BaseFeatureSampler):
        """Rename the same expression on each trial to exercise semantic deduplication."""

        def sample(self, context):
            """Return the ratio under a trial-specific display alias.

            Args:
                context (SearchContext): Schema, current parent, inner-trial evidence and remaining budgets.

            Returns:
                Proposal: Numerically identical expression named using context.number.
            """
            return Proposal(features=[feature(name=f"ratio_{context.number}")])

    X, y, schema = dataset()
    ev = CVEvaluator(LogisticRegression(max_iter=1000), cv=2, importance_repeats=1)
    study = create_study(sampler=AliasSampler(), search_strategy="independent", storage=tmp_path, cache=False)
    study.optimize(X, y, schema, evaluator=ev, n_trials=2, refit=False)
    assert study.trials[2].state == "COMPLETE" and study.trials[2].model_fits == 2
    assert "ratio_2" in study.trials[2].importance and "ratio_1" not in study.trials[2].importance
    resumed = load_study(tmp_path, sampler=AliasSampler())
    resumed.optimize(X, y, schema, evaluator=ev, n_trials=1, refit=False)
    assert resumed.trials[-1].state == "COMPLETE"
    ev.cache = ArtifactCache()
    first = [feature(), feature("square", "square", ["ratio"])]
    renamed = [feature("alias"), feature("renamed", "square", ["alias"])]
    a = ev.evaluate(X, y, schema, first)
    b = ev.evaluate(X, y, schema, renamed)
    assert not b.cache_hit and b.model_fits == 2 and a.fold_values == b.fold_values
    assert "renamed" in b.importance and "square" not in b.importance
    assert (
        FeatureSet.from_features(schema, first).expression_hash
        == FeatureSet.from_features(schema, renamed).expression_hash
    )


def test_dataframe_estimator_names_change_score_even_with_cache():
    """Keep name-sensitive estimator scores separate despite equal expressions."""
    import numpy as np

    from featune import DatasetSchema, FeatureSpec, FieldSchema, Hypothesis

    X = pd.DataFrame({"x": np.arange(1.0, 11.0)})
    y = X.x.to_numpy()
    schema = DatasetSchema(fields=[FieldSchema(name="x", description="x")])
    ev = CVEvaluator(NameModel(), metric="rmse", cv=2, importance_repeats=0, cache=ArtifactCache())
    for name, expected in [("good", 0.0), ("bad", None)]:
        candidate = FeatureSpec(
            name=name, op="abs", inputs=["x"], hypothesis=Hypothesis(concept="x", rationale="x")
        )
        result = ev.evaluate(X, y, schema, [candidate])
        assert not result.cache_hit
        if expected is None:
            assert result.value > 6
        else:
            assert result.value == expected


def test_budget_normal_stop_and_private_statistics():
    """Verify zero budgets stop normally and summary statistics disclose no category labels.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    X, y, schema = dataset()
    study = create_study(budget=SearchBudget(max_model_fits=0))
    study.optimize(X, y, schema, evaluator=evaluator(), n_trials=5)
    assert not study.trials and study.stop_reason == "max_model_fits"
    study = create_study(sampler=RatioSampler(), budget=SearchBudget(max_valid_proposals=1))
    study.optimize(X, y, schema, evaluator=evaluator(), n_trials=5)
    assert len(study.trials) == 2 and study.stop_reason == "max_valid_proposals"
    X["region"] = "private-category"
    stats = study._column_statistics(X, schema)
    assert "private-category" not in json.dumps(stats)
    assert "target" not in stats and "quantiles" in stats["loan"]
    with pytest.raises(ValueError, match="joint_optimization"):
        create_study(param_space={"C": [1, 2]})


def test_evaluator_rejects_changed_data_before_cache_lookup():
    """Verify changed targets/features cannot reuse stale evaluation state without prepare.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    X, y, schema = dataset()
    ev = evaluator()
    ev.cache = ArtifactCache()
    ev.evaluate(X, y, schema, [])
    changed = 1 - y
    with pytest.raises(ValueError, match="call prepare"):
        ev.evaluate(X, changed, schema, [])
    with pytest.raises(ValueError, match="call prepare"):
        ev.evaluate(X.assign(loan=X.loan + 1), y, schema, [])
    ev.prepare(X, changed)
    assert not ev.evaluate(X, changed, schema, []).cache_hit
    assert ev.evaluate(X, changed, schema, []).cache_hit


def test_cost_budget_preflight_and_unknown_cost():
    """Verify cost reservation, known charges and rejection of non-finite token prices.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    calls = []

    def handler(request):
        """Record requests and return known usage for an exact cost-budget boundary.

        Args:
            request (httpx.Request): Mock client request; never forwarded to an external provider.

        Returns:
            httpx.Response: HTTP 200 completion with two input and two output tokens.
        """
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "{}"}],
                "usage": {"input_tokens": 2, "output_tokens": 2},
            },
        )

    client = LLMClient(
        api_key="test",
        max_tokens=2,
        input_cost_per_million=1,
        output_cost_per_million=1,
        cost_budget=0.000004,
        transport=httpx.MockTransport(handler),
    )
    client.complete("hi")
    with pytest.raises(BudgetExceeded):
        client.complete("hi")
    assert len(calls) == 1 and client.cost == 0.000004
    assert LLMClient(api_key="test").cost is None
    with pytest.raises(ValueError):
        LLMClient(api_key="test", input_cost_per_million=float("nan"))


def test_efficiency_targets_and_pareto():
    """Verify resource-to-target summaries, Pareto filtering and unknown billing propagation.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    records = [
        {
            "number": i,
            "state": "COMPLETE",
            "value": score,
            "model_fits": 2,
            "valid_proposal": bool(i),
            "elapsed": i + 1,
            "api_cost": 0.0,
        }
        for i, score in enumerate([0.8, 0.81, 0.805, 0.82])
    ]
    summary, curve = search_diagnostics(records, target_gain=0.005)
    assert summary["candidate_evaluations"] == 3
    assert summary["fixed_gain"]["evaluations_to_X"] == 1
    assert summary["95pct_final_gain"]["evaluations_to_X"] == 3
    assert pareto_frontier(curve).trial.tolist() == [0, 1, 3]
    records[1]["tokens"] = {"unknown_requests": 1}
    assert search_diagnostics(records)[0]["api_cost"] is None


def test_description_ablation_preserves_values():
    """Verify semantic ablations change descriptions/names without changing data or targets.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    from benchmarks.datasets import load_dataset

    original, y, schema = load_dataset("diabetes")
    shuffled, shuffled_y, shuffled_schema = load_dataset("diabetes", shuffled=True, seed=22)
    anonymous, anonymous_y, anonymous_schema = load_dataset("diabetes", anonymous=True)
    assert original.equals(shuffled) and y.equals(shuffled_y) and y.equals(anonymous_y)
    assert (original.to_numpy() == anonymous.to_numpy()).all()
    assert all(a.description != b.description for a, b in zip(schema.fields, shuffled_schema.fields))
    assert all(field.name.startswith("x") for field in anonymous_schema.fields)


def test_covtype_schema_contract(monkeypatch):
    """Verify CoverType loader shape/type semantics using an in-memory fetch fixture.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture restoring patched dataset readers or dependencies.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    from benchmarks import datasets

    class Fake:
        """Stand in for the sklearn CoverType dataset bundle without network access.

        Attributes:
            data (pandas.DataFrame): Two numeric environmental columns.
            target (pandas.Series): Two example multiclass labels.
        """

        data = pd.DataFrame({"Elevation": [1, 2], "Wilderness_Area1": [0, 1]})
        target = pd.Series([1, 7])

    monkeypatch.setattr(datasets, "fetch_covtype", lambda **kwargs: Fake())
    X, y, schema = datasets.load_dataset("covtype", data_dir="/tmp/featune-test-covtype")
    assert X.shape == (2, 2) and y.nunique() == 2 and schema.objective.startswith("Predict")
    assert all(field.dtype == "numeric" for field in schema.fields)


def test_retail_target_does_not_depend_on_other_rows(tmp_path, monkeypatch):
    """Verify changing a held-out target cannot alter a different row through global clipping.

    Args:
        tmp_path (pathlib.Path): Pytest-managed temporary directory isolating persisted artifacts.
        monkeypatch (pytest.MonkeyPatch): Fixture restoring patched dataset readers or dependencies.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    import zipfile

    from benchmarks.datasets import load_dataset

    with zipfile.ZipFile(tmp_path / "online-retail-ii.zip", "w") as archive:
        archive.writestr("online_retail_II.xlsx", b"mock workbook")
    frame = pd.DataFrame(
        {
            "Quantity": range(1, 101),
            "Price": 1.0,
            "InvoiceDate": pd.Timestamp("2020-01-01"),
            "StockCode": "sku",
            "Country": "UK",
        }
    )
    monkeypatch.setattr(
        pd,
        "read_excel",
        lambda *args, **kwargs: (
            frame.copy() if kwargs["sheet_name"] == "Year 2009-2010" else frame.iloc[:0].copy()
        ),
    )
    _, original, _ = load_dataset("online_retail", data_dir=tmp_path)
    frame.loc[99, "Quantity"] = 1
    _, changed, _ = load_dataset("online_retail", data_dir=tmp_path)
    pd.testing.assert_series_equal(original.iloc[:99], changed.iloc[:99])
    assert original.iloc[-1] == 100


def test_wall_time_and_llm_budget_stop_before_work():
    """Verify exhausted time/token/cost settings stop or reject work before unintended spending.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    X, y, schema = dataset()
    study = create_study(budget=SearchBudget(max_wall_time=1e-12))
    study.optimize(X, y, schema, evaluator=evaluator(), n_trials=1)
    assert study.trials[0].state == "BUDGET_EXCEEDED" and study.consumption["model_fits"] == 0
    client = LLMClient(
        api_key="test", transport=httpx.MockTransport(lambda request: pytest.fail("No request expected"))
    )
    study = create_study(sampler=LLMSampler(client=client), budget=SearchBudget(max_llm_tokens=0))
    study.optimize(X, y, schema, evaluator=evaluator(), n_trials=1)
    assert study.stop_reason == "max_llm_tokens" and client.usage.requests == 0
    study = create_study(sampler=LLMSampler(client=client), budget=SearchBudget(max_llm_cost=1))
    with pytest.raises(ValueError, match="prices"):
        study.optimize(X, y, schema, evaluator=evaluator(), n_trials=1)
