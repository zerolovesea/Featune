# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""CLI, notebook and optional-dependency acceptance.

Exercises the module contracts with deterministic local fixtures. External LLM
requests are mocked; temporary storage keeps acceptance runs isolated.

Created:
    2026-09-21
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from test_core import dataset

from featune import CVEvaluator, RandomSampler, create_study
from featune.cli import build_sampler, main


def test_cli_round_trip(tmp_path, capsys):
    """Verify configuration-driven optimize, inspect, feature export and resumed attempts.

    Args:
        tmp_path (pathlib.Path): Pytest-managed temporary directory isolating persisted artifacts.
        capsys (pytest.CaptureFixture): Fixture capturing command output for assertions.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    X, y, schema = dataset()
    X.assign(target=y).to_csv(tmp_path / "data.csv", index=False)
    config = {
        "data": {"path": "data.csv", "target": "target"},
        "schema": schema.model_dump(),
        "estimator": {"type": "logistic"},
        "study": {"storage": "output"},
        "evaluation": {"cv": 2, "importance_repeats": 0},
        "optimize": {"n_trials": 1},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    main(["optimize", str(path)])
    assert (tmp_path / "output/default/pipeline.joblib").exists()
    main(["inspect", "--storage", str(tmp_path / "output")])
    exported = tmp_path / "best.json"
    main(["export", "--storage", str(tmp_path / "output"), "--output", str(exported)])
    assert "features" in json.loads(exported.read_text())
    assert "COMPLETE" in capsys.readouterr().out
    main(["optimize", str(path)])
    trials = json.loads((tmp_path / "output/default/trials.json").read_text())
    assert len(trials) == 3


def test_inline_secrets_rejected():
    """Verify CLI sampler configuration rejects embedded API credentials.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    for config in [{"type": "llm", "api_key": "secret"}, {"type": "hybrid", "llm": {"api_key": "secret"}}]:
        with pytest.raises(ValueError, match="FEATUNE_API_KEY"):
            build_sampler(config)


def test_refit_false_does_not_export_a_stale_pipeline():
    """Verify refit=False cannot expose a previously fitted model as the current result.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    X, y, schema = dataset()
    evaluator = CVEvaluator(LogisticRegression(), cv=2, importance_repeats=0)
    study = create_study(sampler=RandomSampler()).optimize(X, y, schema, evaluator=evaluator, n_trials=0)
    assert study.export_pipeline() is not None
    study.optimize(X, y, schema, evaluator=evaluator, n_trials=1, refit=False)
    with pytest.raises(ValueError, match="refit=True"):
        study.export_pipeline()


def test_readme_python_quickstarts(monkeypatch):
    # Execute the self-contained quickstart block in both documentation languages.
    """Execute documented quickstarts with a local estimator replacing the weight download.

    Args:
        monkeypatch (pytest.MonkeyPatch): Isolates the default model factory from the network.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    monkeypatch.setattr(
        "featune.evaluation.default_estimator",
        lambda metric, random_state: LogisticRegression(max_iter=1000, random_state=random_state),
    )
    root = Path(__file__).resolve().parents[1]
    for name in ["README.md", "README_en.md"]:
        match = re.search(
            r"^(`{3}|~{3})python\n(.*?)^\1\s*$", (root / name).read_text(), re.MULTILINE | re.DOTALL
        )
        assert match is not None, f"Missing Python quickstart in {name}"
        block = match.group(2)
        scope = {}
        exec(compile(block, name, "exec"), scope)
        assert np.isfinite(scope["study"].best_value)


def test_cli_budget_stop_without_baseline(tmp_path, capsys):
    """Verify zero-fit CLI runs emit a normal stop without exporting a nonexistent model.

    Args:
        tmp_path (pathlib.Path): Pytest-managed temporary directory isolating persisted artifacts.
        capsys (pytest.CaptureFixture): Fixture capturing command output for assertions.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    X, y, schema = dataset()
    X.assign(target=y).to_csv(tmp_path / "data.csv", index=False)
    config = {
        "data": {"path": "data.csv", "target": "target"},
        "schema": schema.model_dump(),
        "study": {"storage": "output", "budget": {"max_model_fits": 0}},
        "evaluation": {"cv": 2},
        "report": True,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    main(["optimize", str(path)])
    result = json.loads(capsys.readouterr().out)
    assert result["best_value"] is None and result["stop_reason"] == "max_model_fits"
    assert not (tmp_path / "output/default/pipeline.joblib").exists()
