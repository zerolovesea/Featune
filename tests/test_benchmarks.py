# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Checks for paired benchmark aggregation and immutable post-hoc evidence.

Created:
    2026-09-22
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from featune import DatasetSchema, FieldSchema
from featune.compiler import build_pipeline


def test_paired_interval():
    """Check directional means, deterministic intervals and small-sample disclosure.

    Returns:
        None: Bootstrap output has the expected scale and minimum-sample behavior.
    """
    pytest.importorskip("matplotlib")
    from benchmarks.summarize_tabpfn import interval

    assert interval([2, 2, 2, 2, 2]) == (2.0, 2.0, 2.0)
    assert interval([-1, 0, 1, 2, 3]) == interval([-1, 0, 1, 2, 3])
    mean, low, high = interval([2])
    assert mean == 2 and np.isnan(low) and np.isnan(high)


def test_posthoc_explanation_keeps_model_frozen(tmp_path, monkeypatch):
    """Ensure permutation evidence preserves inputs and never refits the selected model.

    Args:
        tmp_path (pathlib.Path): Isolated evidence output directory.
        monkeypatch (pytest.MonkeyPatch): Adds the benchmark module lookup path.

    Returns:
        None: Known predictive input has positive reliance and predictions stay unchanged.
    """
    directory = Path(__file__).resolve().parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("featune_benchmark_runner", directory / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    X = pd.DataFrame({"x": np.arange(30, dtype=float)})
    y = X.x * 3
    schema = DatasetSchema(fields=[FieldSchema(name="x", description="Predictive input")])
    pipeline = build_pipeline(schema, [], Ridge()).fit(X, y)
    before = pipeline.predict(X)
    original = X.copy(deep=True)
    result = module.explain_selected(pipeline, X, y, ["x"], "rmse", 42, tmp_path)
    assert result["positive_reliance_features"] == 1
    np.testing.assert_array_equal(before, pipeline.predict(X))
    pd.testing.assert_frame_equal(original, X)
