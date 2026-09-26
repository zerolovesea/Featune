# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Contracts, leakage boundaries and end-to-end search acceptance.

Exercises the module contracts with deterministic local fixtures. External LLM
requests are mocked; temporary storage keeps acceptance runs isolated.

Created:
    2026-09-21
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit

from featune import (
    CVEvaluator,
    DatasetSchema,
    FeatureCompiler,
    FeatureSpec,
    FieldSchema,
    Hypothesis,
    Proposal,
    RandomSampler,
    create_study,
    load_study,
)
from featune.evaluation import EvaluationResult
from featune.samplers import BaseFeatureSampler, EvolutionSampler, SearchContext
from featune.schema import validate_features
from featune.storage import Storage
from featune.study import Trial


def dataset():
    """Create deterministic loan/income/category fixtures for search tests.

    Returns:
        tuple[pandas.DataFrame, pandas.Series, DatasetSchema]: Aligned mixed-type rows, binary labels
        and semantics.
    """
    rng = np.random.default_rng(11)
    X = pd.DataFrame(
        {
            "loan": rng.uniform(1, 20, 120),
            "income": rng.uniform(1, 12, 120),
            "region": rng.choice(["a", "b", "c"], 120),
        }
    )
    y = (X.loan / X.income > 1.8).astype(int)
    schema = DatasetSchema(
        fields=[
            FieldSchema(name=name, description=name, dtype="categorical" if name == "region" else "numeric")
            for name in X
        ]
    )
    return X, y, schema


def test_loss_direction_defaults_and_conflicts():
    """Choose the lower loss by default and reject contradictory directions."""
    study = create_study(metric="rmse")
    assert study.direction == "minimize"
    study.trials = [
        Trial(number=0, state="COMPLETE", value=1.0),
        Trial(number=1, state="COMPLETE", value=2.0),
    ]
    assert study.best_trial.number == 0
    with pytest.raises(ValueError, match="requires direction"):
        create_study(metric="rmse", direction="maximize")


def test_conflicting_hypothesis_direction_is_inconclusive():
    """Do not claim support when observed and expected directions disagree."""
    study = create_study()
    parent = Trial(number=0, state="COMPLETE", value=0.5, fold_values=[0.5, 0.5])
    study.trials = [parent]
    trial = Trial(
        number=1,
        features=[
            FeatureSpec(
                name="ratio",
                op="divide",
                inputs=["loan", "income"],
                hypothesis=Hypothesis(
                    concept="burden", rationale="Debt relative to income", expected_direction="increasing"
                ),
            ).model_dump()
        ],
    )
    result = EvaluationResult(
        value=0.6,
        metrics={"auc": 0.6},
        fold_values=[0.6, 0.6],
        importance={"ratio": 0.2},
        directions={"ratio": "decreasing"},
    )
    study._complete(trial, result, parent)
    assert trial.hypotheses["ratio"]["status"] == "inconclusive"
    assert trial.hypotheses["ratio"]["reason"] == "direction_conflict"


def test_load_study_rejects_path_before_creating_storage(tmp_path):
    """Reject traversal without creating a database outside the study root."""
    with pytest.raises(ValueError, match="single directory"):
        load_study(tmp_path / "root", "../outside")
    assert not (tmp_path / "outside").exists()


def feature(name="ratio", op="divide", inputs=None, **kwargs):
    """Create a validated feature spec used by numerical and rejection tests.

    Args:
        name (str): Derived output identifier.
        op (str): Closed DSL operator under test.
        inputs (list[str] or None): Operator inputs; None/empty uses loan and income.
        kwargs (Any): Additional FeatureSpec overrides, such as clip parameters.

    Returns:
        FeatureSpec: Requested operator with a fixed repayment-burden hypothesis.
    """
    return FeatureSpec(
        name=name,
        op=op,
        inputs=inputs or ["loan", "income"],
        hypothesis=Hypothesis(concept="burden", rationale="Debt relative to income"),
        **kwargs,
    )


class RatioSampler(BaseFeatureSampler):
    """Propose the same loan/income ratio to exercise duplicate-result reuse."""

    def sample(self, context):
        """Return the deterministic repayment-ratio fixture.

        Args:
            context (SearchContext): Schema, current parent, inner-trial evidence and remaining budgets.

        Returns:
            Proposal: One ratio FeatureSpec; the context is intentionally not inspected.
        """
        return Proposal(features=[feature()])


@pytest.mark.parametrize(
    "changes",
    [
        {"op": "eval"},
        {"inputs": ["loan"]},
        {"params": {"epsilon": 0.1}},
        {"name": "__import__('os')"},
        {"op": "clip", "inputs": ["loan"], "params": {"lower": 3.0, "upper": 1.0}},
        {"op": "clip", "inputs": ["loan"], "params": {"lower": float("nan"), "upper": 3.0}},
    ],
)
def test_invalid_dsl(changes):
    """Verify invalid operators, names, arity and parameter bounds fail schema validation.

    Args:
        changes (dict): Invalid FeatureSpec field overrides for this parameterized rejection case.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    data = feature().model_dump()
    data.update(changes)
    with pytest.raises(ValueError):
        FeatureSpec.model_validate(data)


@pytest.mark.parametrize(
    "candidate",
    [
        feature(inputs=["target", "income"]),
        feature(inputs=["ratio", "income"]),
        feature(inputs=["region", "income"]),
        feature(name="income"),
    ],
)
def test_references_and_types(candidate):
    """Verify unknown, forward, mismatched and reserved input/output references are rejected.

    Args:
        candidate (FeatureSpec): Invalid reference/type/name candidate expected to be rejected.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    _, _, schema = dataset()
    with pytest.raises(ValueError):
        validate_features(schema, [candidate])


def test_excluded_fields_and_duplicate_expressions():
    """Verify excluded columns never enter compilation and repeated expressions are rejected.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    X, _, schema = dataset()
    schema = schema.model_copy(
        update={"fields": schema.fields + [FieldSchema(name="future", description="future", exclude=True)]}
    )
    with pytest.raises(ValueError):
        validate_features(schema, [feature(inputs=["future", "income"])])
    with pytest.raises(ValueError):
        validate_features(schema, [feature(), feature(name="another")])
    X["future"] = 100
    assert "future" not in FeatureCompiler(schema).fit_transform(X)


def test_statistics_fit_only_on_training_and_unknown_categories():
    """Verify training-only group/frequency maps and deterministic unseen-category fallbacks.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    _, _, schema = dataset()
    train = pd.DataFrame({"loan": [2.0, 4.0, 9.0], "income": [0.0, 2.0, 3.0], "region": ["a", "a", "b"]})
    test = pd.DataFrame({"loan": [1000.0, 500.0], "income": [0.0, 1.0], "region": ["a", "new"]})
    features = [
        feature(),
        feature("mean", "group_mean", ["region", "loan"]),
        feature("freq", "frequency", ["region"]),
    ]
    compiler = FeatureCompiler(schema, features).fit(train)
    result = compiler.transform(test)
    assert result["mean"].tolist() == [3.0, 5.0]
    assert result["freq"].tolist() == [2 / 3, 0.0]
    assert pd.isna(result.ratio.iloc[0])
    assert compiler.transform(test.iloc[:1])["mean"].iloc[0] == result["mean"].iloc[0]


def test_dates_missing_and_categorical_cross():
    """Verify datetime arithmetic, missingness and canonical categorical cross encoding.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    X = pd.DataFrame(
        {"a": ["2020-01-01", None], "b": ["2019-12-31", "2020-01-02"], "c": [None, "N:"], "d": ["x", "y"]}
    )
    schema = DatasetSchema(
        fields=[
            FieldSchema(name=n, description=n, dtype=("datetime" if n in "ab" else "categorical")) for n in X
        ]
    )
    features = [
        feature("days", "days_between", ["a", "b"]),
        feature("year", "year", ["a"]),
        feature("missing", "is_missing", ["a"]),
        feature("pair", "cross", ["c", "d"]),
    ]
    output = FeatureCompiler(schema, features).fit_transform(X)
    assert output.days.iloc[0] == 1 and output.year.iloc[0] == 2020
    assert output.missing.tolist() == [0.0, 1.0]
    assert output.c.iloc[0] != output.c.iloc[1]
    assert output["pair"].nunique() == 2
    reversed_pair = FeatureCompiler(schema, [feature("pair", "cross", ["d", "c"])]).fit_transform(X)
    pd.testing.assert_series_equal(output["pair"], reversed_pair["pair"])
    mixed = pd.Series([1, "1", None])
    from featune.compiler import category_key

    assert len(set(category_key(mixed))) == 3


def test_cv_export_and_resume(tmp_path):
    """Verify duplicate reuse, exported predictions and incompatible-data resume rejection.

    Args:
        tmp_path (pathlib.Path): Pytest-managed temporary directory isolating persisted artifacts.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    X, y, schema = dataset()
    model = LogisticRegression(max_iter=1000)
    study = create_study(sampler=RatioSampler(), search_strategy="independent", storage=tmp_path)
    study.optimize(X, y, schema, evaluator=CVEvaluator(model, cv=3), n_trials=2)
    assert [t.state for t in study.trials] == ["COMPLETE", "COMPLETE", "DUPLICATE"]
    assert study.best_value >= study.baseline_trial.value
    assert study.feature_history().iloc[0]["concept"] == "burden"
    output = tmp_path / "pipeline.joblib"
    study.export_pipeline(output)
    np.testing.assert_allclose(study.pipeline_.predict_proba(X), joblib.load(output).predict_proba(X))
    resumed = load_study(tmp_path, sampler=RatioSampler())
    assert resumed.stop_reason == "n_trials"
    resumed.optimize(X, y, schema, evaluator=CVEvaluator(model, cv=3), n_trials=1)
    assert len(resumed.trials) == 4 and resumed.trials[-1].state == "DUPLICATE"
    with pytest.raises(ValueError, match="Resume rejected"):
        resumed.optimize(X.assign(loan=X.loan + 1), y, schema, evaluator=CVEvaluator(model, cv=3), n_trials=0)


def test_deterministic_restart_and_beam(tmp_path):
    """Verify chunked resume matches uninterrupted search and expands multiple beam parents.

    Args:
        tmp_path (pathlib.Path): Pytest-managed temporary directory isolating persisted artifacts.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    X, y, schema = dataset()
    kwargs = dict(sampler=RandomSampler(12), search_strategy="beam", beam_width=2)

    def evaluator():
        """Construct a cheap deterministic evaluator for recovery/cache acceptance.

        Returns:
            CVEvaluator: Two-fold logistic classifier with permutation explanation disabled.
        """
        return CVEvaluator(LogisticRegression(max_iter=1000), cv=2, importance_repeats=0)

    full = create_study(**kwargs).optimize(X, y, schema, evaluator=evaluator(), n_trials=6)
    part = create_study(storage=tmp_path, **kwargs).optimize(X, y, schema, evaluator=evaluator(), n_trials=3)
    part = load_study(tmp_path, sampler=RandomSampler(12)).optimize(
        X, y, schema, evaluator=evaluator(), n_trials=3
    )
    assert [(t.signature, t.parent, t.value) for t in full.trials] == [
        (t.signature, t.parent, t.value) for t in part.trials
    ]
    assert len({t.parent for t in full.trials[3:5]}) == 2


def test_minimize_hyperparameters_callbacks_and_failure():
    """Verify loss minimization, finite HPO choices, callback stopping and failed proposals.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    X, y, schema = dataset()
    study = create_study(
        metric="rmse",
        direction="minimize",
        sampler=RandomSampler(),
        joint_optimization=True,
        param_space={"alpha": [0.1, 1.0]},
    )
    study.optimize(
        X,
        y.astype(float),
        schema,
        evaluator=CVEvaluator(Ridge(), metric="rmse", cv=3, importance_repeats=0),
        n_trials=10,
        callbacks=[lambda study, trial: study.stop()],
    )
    assert len(study.trials) == 2
    assert study.best_value <= study.baseline_trial.value
    assert study.trials[-1].model_params["alpha"] in [0.1, 1.0]

    class BadSampler(BaseFeatureSampler):
        """Inject an unknown raw-field reference to exercise failed-trial recording."""

        def sample(self, context):
            """Return a syntactically valid proposal with an unknown input field.

            Args:
                context (SearchContext): Schema, current parent, inner-trial evidence and remaining budgets.

            Returns:
                Proposal: Candidate expected to fail study reference validation.
            """
            return Proposal(features=[feature(inputs=["unknown", "income"])])

    broken = create_study(sampler=BadSampler()).optimize(
        X, y, schema, estimator=LogisticRegression(), n_trials=1
    )
    assert broken.trials[-1].state == "FAIL"


def test_groups_and_time_boundaries():
    """Verify group/time split isolation and rejection of invalid overlapping folds.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    X, y, _ = dataset()
    groups = np.repeat(np.arange(30), 4)
    evaluator = CVEvaluator(LogisticRegression(), cv=3).prepare(X, y, groups)
    assert all(not set(groups[a]) & set(groups[b]) for a, b in evaluator.splits_)
    with pytest.raises(ValueError, match="Groups overlap"):
        CVEvaluator(LogisticRegression(), cv=KFold(3, shuffle=True, random_state=42)).prepare(X, y, groups)
    timed = CVEvaluator(Ridge(), metric="rmse", cv=TimeSeriesSplit(3)).prepare(X, y)
    assert all(a.max() < b.min() for a, b in timed.splits_)
    with pytest.raises(ValueError, match="overlapping"):
        CVEvaluator(Ridge(), metric="rmse", cv=[([0, 1], [1, 2]), ([1], [3])]).prepare(X, y)


def test_storage_lock_and_interruption(tmp_path):
    """Verify single-writer exclusion and interrupted-trial recovery without ID reuse.

    Args:
        tmp_path (pathlib.Path): Pytest-managed temporary directory isolating persisted artifacts.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    storage = Storage(tmp_path)
    with storage.lock():
        with pytest.raises(RuntimeError, match="Another writer"):
            with Storage(tmp_path).lock():
                pass
    X, y, schema = dataset()
    study = create_study(storage=tmp_path, sampler=RandomSampler()).optimize(
        X, y, schema, evaluator=CVEvaluator(LogisticRegression(), cv=2, importance_repeats=0), n_trials=0
    )
    from featune.study import Trial

    study.storage.save(Trial(number=1).to_dict())
    study.optimize(
        X, y, schema, evaluator=CVEvaluator(LogisticRegression(), cv=2, importance_repeats=0), n_trials=1
    )
    assert study.trials[1].state == "INTERRUPTED" and study.trials[2].number == 2


def test_samplers_produce_controlled_features():
    """Verify random/evolution proposals satisfy the closed DSL across deterministic trials.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    _, _, schema = dataset()
    for sampler in (RandomSampler(features_per_trial=3), EvolutionSampler(features_per_trial=2)):
        for number in range(1, 8):
            proposal = sampler.sample(SearchContext(schema, number, [], [], "auc", "maximize"))
            validate_features(schema, proposal.features)
            assert proposal.features


def test_report_escapes_user_text(tmp_path):
    """Verify report rendering escapes user-provided text and includes evidence tables.

    Args:
        tmp_path (pathlib.Path): Pytest-managed temporary directory isolating persisted artifacts.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    pytest.importorskip("plotly")
    X, y, schema = dataset()
    study = create_study(study_name="<script>alert(1)<script>").optimize(
        X, y, schema, estimator=LogisticRegression(), n_trials=0
    )
    report = study.report(tmp_path / "report.html")
    assert "<script>alert(1)<script>" not in report
    assert "&lt;script&gt;alert(1)&lt;script&gt;" in report
    assert "Concept evidence" in report


@pytest.mark.parametrize(
    "op,inputs,params,expected",
    [
        ("add", ["loan", "income"], {}, [5.0, 10.0]),
        ("subtract", ["loan", "income"], {}, [1.0, 6.0]),
        ("multiply", ["loan", "income"], {}, [6.0, 16.0]),
        ("abs", ["loan"], {}, [3.0, 8.0]),
        ("square", ["income"], {}, [4.0, 4.0]),
        ("sqrt_abs", ["loan"], {}, [np.sqrt(3), np.sqrt(8)]),
        ("log1p_abs", ["loan"], {}, [np.log(4), np.log(9)]),
        ("clip", ["loan"], {"lower": 4.0, "upper": 6.0}, [4.0, 6.0]),
        ("group_std", ["region", "loan"], {}, [2.5, 2.5]),
    ],
)
def test_operator_numerics(op, inputs, params, expected):
    """Verify each parameterized operator produces its expected fixture values.

    Args:
        op (str): Closed DSL operator under test.
        inputs (list[str]): Raw or already-derived field names consumed by the operator.
        params (dict[str, float]): Operator parameters for the numerical fixture.
        expected (sequence[float]): Expected transformed numerical values for the fixture rows.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    _, _, schema = dataset()
    X = pd.DataFrame({"loan": [3.0, 8.0], "income": [2.0, 2.0], "region": ["a", "a"]})
    transformed = FeatureCompiler(schema, [feature("result", op, inputs, params=params)]).fit_transform(X)
    np.testing.assert_allclose(transformed.result, expected)


def test_multiclass_and_loss_metrics():
    """Verify multiclass AUC and additional loss/F1 metrics share valid evaluation folds.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    schema = DatasetSchema(fields=[FieldSchema(name=n, description=n) for n in data.data])
    study = create_study(metric="auc_ovr").optimize(
        data.data,
        data.target,
        schema,
        evaluator=CVEvaluator(
            LogisticRegression(max_iter=1000),
            metric="auc_ovr",
            metrics=["log_loss", "f1"],
            cv=3,
            importance_repeats=0,
        ),
        n_trials=1,
    )
    assert study.best_value > 0.9 and study.best_trial.metrics["log_loss"] > 0


@pytest.mark.parametrize(
    "dtype,values", [("int8", [20, -128]), ("int64", [4_000_000_000, -4_000_000_000]), ("uint8", [0, 255])]
)
def test_integer_arithmetic_is_promoted_before_compilation(dtype, values):
    """Verify signed/unsigned integer arithmetic is promoted before overflow can occur.

    Args:
        dtype (str): Integer dtype exercising signed overflow or unsigned subtraction.
        values (sequence[int]): Boundary input values that would wrap without floating promotion.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    X = pd.DataFrame(
        {
            "loan": pd.Series(values, dtype=dtype),
            "income": pd.Series([2, 3], dtype=dtype),
            "region": ["a", "b"],
        }
    )
    _, _, schema = dataset()
    result = FeatureCompiler(
        schema,
        [
            feature("squared", "square", ["loan"]),
            feature("difference", "subtract"),
            feature("absolute", "abs", ["loan"]),
        ],
    ).fit_transform(X)
    expected = X.loan.astype(float)
    np.testing.assert_allclose(result.squared, expected**2)
    np.testing.assert_allclose(result.difference, expected - X.income.astype(float))
    np.testing.assert_allclose(result.absolute, expected.abs())
