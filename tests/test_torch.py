# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Optional tabular-network and embedding integration checks.

Exercises the module contracts with deterministic local fixtures. External LLM
requests are mocked; temporary storage keeps acceptance runs isolated.

Created:
    2026-09-21
"""

import numpy as np
import pytest
from sklearn.base import clone, is_classifier

pytest.importorskip("torch")
from test_core import dataset

from featune import CVEvaluator, RandomSampler, create_study
from featune.torch import TorchClassifier, TorchRegressor


def test_torch_classification_and_unknown_embeddings():
    """Verify seeded classifier fitting, unseen category handling and approved parameter search.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    X, y, schema = dataset()
    estimator = TorchClassifier(hidden_dims=(8,), epochs=2, batch_size=32, embedding_dim=3)
    assert is_classifier(estimator)
    model = clone(estimator).fit(X, y)
    test = X.iloc[:3].copy()
    test["region"] = "unseen"
    probabilities = model.predict_proba(test)
    assert probabilities.shape == (3, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(model.predict_proba(X), clone(estimator).fit(X, y).predict_proba(X))
    study = create_study(
        sampler=RandomSampler(),
        joint_optimization=True,
        param_space={"learning_rate": [0.01], "embedding_dim": [2], "hidden_dims": [[8], [8, 4]]},
    )
    study.optimize(X, y, schema, evaluator=CVEvaluator(estimator, cv=2, importance_repeats=0), n_trials=1)
    assert study.trials[1].state == "COMPLETE"
    assert study.trials[1].model_params["embedding_dim"] == 2


def test_torch_regression():
    """Verify the optional regressor returns finite predictions in target units.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    X, y, _ = dataset()
    estimator = TorchRegressor(hidden_dims=(8,), epochs=2, batch_size=32).fit(X, y)
    prediction = estimator.predict(X.iloc[:4])
    assert prediction.shape == (4,) and np.isfinite(prediction).all()
