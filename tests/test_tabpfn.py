# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Default-model contracts and opt-in real-checkpoint integration acceptance.

Ordinary checks require no downloads. Set FEATUNE_TEST_TABPFN=1 to exercise
classification, regression, fold caches and fresh-process artifact restoration.

Created:
    2026-09-22
"""

import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.linear_model import Ridge

from featune import CVEvaluator, DatasetSchema, FieldSchema, TabPFNClassifier, TabPFNRegressor, create_study
from featune.cli import build_estimator, main
from featune.compiler import build_pipeline
from featune.tabpfn import PRESETS, TABPFN_VERSION, V35_WEIGHTS, _checkpoint


def test_default_task_selection_and_pipeline():
    """Check metric-driven defaults, seeds, cloning and native DataFrame preprocessing.

    Returns:
        None: Assertions fail if task selection or pipeline routing regresses.
    """
    for metric in ["auc", "auc_ovr", "accuracy", "f1", "log_loss", "rmse", "mae", "r2"]:
        evaluator = CVEvaluator(metric=metric, random_state=17)
        classifier = metric not in {"rmse", "mae", "r2"}
        assert is_classifier(evaluator.estimator) == classifier
        assert is_regressor(evaluator.estimator) != classifier
        assert evaluator.estimator.random_state == 17
        assert evaluator.importance_repeats == 0
        assert clone(evaluator.estimator).provenance() == evaluator.estimator.provenance()
        assert type(build_estimator({}, metric)) is type(evaluator.estimator)
    assert CVEvaluator().estimator.model_version == "v2"
    with pytest.raises(ValueError, match="model_version"):
        TabPFNClassifier(model_version="latest").provenance()
    assert CVEvaluator(Ridge(), metric="rmse").importance_repeats == 2
    assert CVEvaluator(importance_repeats=1).importance_repeats == 1
    with pytest.raises(ValueError, match="n_jobs=1"):
        CVEvaluator(importance_repeats=1, n_jobs=2)
    schema = DatasetSchema(
        fields=[FieldSchema(name="kind", dtype="categorical", description="Input category")]
    )
    pipeline = build_pipeline(schema, [], TabPFNClassifier())
    assert list(pipeline.named_steps) == ["features", "model"]
    transformed = pipeline["features"].fit_transform(pd.DataFrame({"kind": ["a", None]}))
    assert list(transformed.columns) == ["kind"]
    assert pd.isna(transformed.iloc[1, 0])
    assert pipeline["model"].provenance()["package"] == TABPFN_VERSION


def test_pinned_checkpoint_integrity(tmp_path, monkeypatch):
    """Reject changed cached weights before executing upstream checkpoint loading.

    Args:
        tmp_path (pathlib.Path): Temporary corrupt checkpoint location.
        monkeypatch (pytest.MonkeyPatch): Replaces the upstream cache lookup without network access.

    Returns:
        None: A mismatched checksum raises RuntimeError.
    """
    path = tmp_path / "checkpoint.ckpt"
    path.write_bytes(b"not the published checkpoint")
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda **kwargs: str(path))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _checkpoint("classifier")
    monkeypatch.setattr("featune.tabpfn.version", lambda package: "0.0.0")
    with pytest.raises(RuntimeError, match="requires tabpfn"):
        _checkpoint("classifier")


def test_capacity_errors_precede_download(monkeypatch):
    """Fail oversized tasks explicitly instead of downloading or silently subsampling.

    Args:
        monkeypatch (pytest.MonkeyPatch): Makes any attempted download an assertion failure.

    Returns:
        None: Model limits fail before loading weights.
    """

    def forbidden(kind, model_version):
        """Reject unexpected checkpoint access.

        Args:
            kind (str): Requested task kind.
            model_version (str): Requested preset.

        Raises:
            AssertionError: Always; invalid inputs must not reach this call.
        """
        raise AssertionError(kind)

    monkeypatch.setattr("featune.tabpfn._checkpoint", forbidden)
    for rows, columns, classes, match in [
        (10001, 1, 2, "10,000"),
        (10, 501, 2, "500"),
        (11, 1, 11, "10 classes"),
        (1001, 1, 2, "CPU"),
    ]:
        with pytest.raises(ValueError, match=match):
            TabPFNClassifier(device="cpu").fit(
                pd.DataFrame(np.zeros((rows, columns))), np.arange(rows) % classes
            )


@pytest.mark.skipif(
    os.getenv("FEATUNE_TEST_TABPFN") != "1", reason="Set FEATUNE_TEST_TABPFN=1 for real weights"
)
@pytest.mark.parametrize("metric", ["auc", "rmse"])
def test_real_fold_cache_and_portable_export(metric, tmp_path):
    """Exercise real mixed-type CV, model isolation, cached resume and offline process reload.

    Args:
        metric (str): Classification or regression metric under test.
        tmp_path (pathlib.Path): Isolated study/export directory.

    Returns:
        None: Real predictions, resource accounting and reloaded predictions match.

    Notes:
        Requires prior license acceptance and access to pinned weights; no LLM calls occur.
    """
    rng = np.random.default_rng(11)
    X = pd.DataFrame({"x": rng.normal(size=40), "kind": ["a", "b", None, "a"] * 10})
    X.loc[0, "x"] = np.nan
    y = np.arange(40) % 2 if metric == "auc" else np.nan_to_num(X.x.to_numpy()) * 2 + 1
    schema = DatasetSchema(
        fields=[
            FieldSchema(name="x", description="Numeric measurement"),
            FieldSchema(name="kind", dtype="categorical", description="Input category"),
        ]
    )
    cls = TabPFNClassifier if metric == "auc" else TabPFNRegressor
    model_version = os.getenv("FEATUNE_TEST_TABPFN_VERSION", "v2")
    evaluator = CVEvaluator(
        cls(model_version=model_version, device="cpu", n_estimators=1), metric=metric, cv=2
    )
    study = create_study(
        metric=metric, direction="maximize" if metric == "auc" else "minimize", storage=tmp_path
    )
    study.optimize(X, y, schema, evaluator=evaluator, n_trials=1)
    assert np.isfinite(study.best_value)
    assert study.baseline_trial.model_fits == 2
    assert study.baseline_trial.prediction_time > 0
    assert (
        study.fingerprint.model_provenance["sha256"]
        == PRESETS[model_version]["classifier" if metric == "auc" else "regressor"]["sha256"]
    )
    assert {"tabpfn", "torch"} <= study.fingerprint.dependencies.keys()
    first = evaluator.evaluate(X, y, schema, [])
    assert first.cache_hit and first.model_fits == 0 and first.prediction_time == 0
    # Simulate a source cache directory that does not exist on the destination.
    study.export_pipeline()["model"].model_.model_path = "/nonexistent/source-cache/model.ckpt"
    pipeline = study.export_pipeline(tmp_path / "pipeline.joblib")
    expected = pipeline.predict(X.iloc[:5])
    restored = joblib.load(tmp_path / "pipeline.joblib")
    assert "model_" not in vars(restored["model"])
    np.testing.assert_allclose(restored.predict(X.iloc[:5]), expected, rtol=1e-6)
    with pytest.raises(ValueError, match="columns"):
        restored["model"].predict(X.iloc[:5][["kind", "x"]])
    # A new process has no fitted tensors and must reconstruct from the persisted archive.
    X.iloc[:5].to_pickle(tmp_path / "input.pkl")
    code = 'import joblib,pandas as pd,json,sys; p=sys.argv[1]; print(json.dumps(joblib.load(p+"/pipeline.joblib").predict(pd.read_pickle(p+"/input.pkl")).tolist()))'
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        check=True,
    )
    np.testing.assert_allclose(json.loads(result.stdout), expected, rtol=1e-6)
    resumed = create_study(metric=metric, direction=study.direction, storage=tmp_path)
    resumed.optimize(
        X,
        y,
        schema,
        evaluator=CVEvaluator(
            cls(model_version=model_version, device="cpu", n_estimators=1), metric=metric, cv=2
        ),
        n_trials=0,
    )
    assert resumed.consumption["model_fits"] == study.consumption["model_fits"]
    np.testing.assert_allclose(resumed.export_pipeline().predict(X.iloc[:5]), expected, rtol=1e-6)

    assert "Built with PriorLabs-TabPFN" in study.report()
    if metric == "rmse":
        X.assign(target=y).to_csv(tmp_path / "data.csv", index=False)
        config = {
            "data": {"path": "data.csv", "target": "target"},
            "schema": schema.model_dump(),
            "study": {"metric": "rmse", "direction": "minimize", "storage": "cli"},
            "estimator": {"params": {"model_version": model_version, "device": "cpu", "n_estimators": 1}},
            "evaluation": {"cv": 2},
            "optimize": {"n_trials": 0},
            "report": True,
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        main(["optimize", str(path)])
        exported = joblib.load(tmp_path / "cli/default/pipeline.joblib")
        assert isinstance(exported["model"], TabPFNRegressor)
        assert np.isfinite(exported.predict(X.iloc[:3])).all()


def test_checkpoint_authorization_and_offline_cache(tmp_path, monkeypatch):
    """Verify pinned downloads require acceptance and offline/cache paths never authenticate.

    Args:
        tmp_path (pathlib.Path): Controlled checkpoint storage.
        monkeypatch (pytest.MonkeyPatch): Isolates networking and upstream authorization.

    Returns:
        None: Download ordering, identity, digest and offline behavior are checked.
    """
    import hashlib

    from huggingface_hub.errors import LocalEntryNotFoundError

    path = tmp_path / V35_WEIGHTS["classifier"]["filename"]
    path.write_bytes(b"controlled weight bytes")
    monkeypatch.setitem(V35_WEIGHTS["classifier"], "sha256", hashlib.sha256(path.read_bytes()).hexdigest())
    calls = []
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda **kwargs: None)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", False)
    monkeypatch.setattr(
        "tabpfn.browser_auth.ensure_license_accepted",
        lambda **kwargs: calls.append(("license", kwargs)),
    )

    def download(**kwargs):
        """Record exact download options and emulate an offline cache miss.

        Args:
            **kwargs: Arguments forwarded to Hugging Face.

        Returns:
            str: Controlled checkpoint filename when online.

        Raises:
            LocalEntryNotFoundError: Offline cache misses cannot download weights.
        """
        calls.append(("download", kwargs))
        if kwargs["local_files_only"]:
            raise LocalEntryNotFoundError("Offline cache miss")
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    assert _checkpoint("classifier", "v3.5") == str(path)
    assert [item[0] for item in calls] == ["license", "download"]
    assert calls[0][1] == {"hf_repo_id": "tabpfn_3_5"}
    assert calls[1][1]["revision"] == V35_WEIGHTS["classifier"]["revision"]
    calls.clear()
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    with pytest.raises(LocalEntryNotFoundError):
        _checkpoint("regressor", "v3.5")
    assert [item[0] for item in calls] == ["download"]
    calls.clear()
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda **kwargs: str(path))
    assert _checkpoint("regressor", "v3.5") == str(path)
    assert not calls


def test_license_failure_stops_download(monkeypatch):
    """Ensure rejected authorization is propagated before any checkpoint download.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces authentication and cache access.

    Returns:
        None: The exact upstream authorization failure is propagated.
    """
    from tabpfn.errors import TabPFNLicenseError

    def reject(**kwargs):
        """Reject authorization without accepting terms for the user.

        Args:
            **kwargs: Upstream repository identity.

        Raises:
            TabPFNLicenseError: User has not accepted the license.
        """
        raise TabPFNLicenseError("Acceptance required")

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda **kwargs: None)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", False)
    monkeypatch.setattr("tabpfn.browser_auth.ensure_license_accepted", reject)
    with pytest.raises(TabPFNLicenseError, match="Acceptance required"):
        _checkpoint("classifier", "v3.5")


@pytest.mark.parametrize("model_version", ["v2", "v3.5"])
@pytest.mark.parametrize("classifier", [True, False])
def test_upstream_task_routing_and_preset_migration(classifier, model_version, monkeypatch):
    """Check both task wrappers forward native inputs and reject obsolete fitted identity.

    Args:
        classifier (bool): Selects classification or regression routing.
        model_version (str): Selected fixed preset.
        monkeypatch (pytest.MonkeyPatch): Replaces only upstream inference and checkpoint I/O.

    Returns:
        None: Parameters, predictions and migration guards match the fixed preset.
    """
    from unittest.mock import Mock

    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import _resolve_model_version

    path = PRESETS[model_version]["classifier"]["filename"]
    assert _resolve_model_version(path) == (ModelVersion.V2 if model_version == "v2" else ModelVersion.V3_5)
    upstream = Mock()
    upstream.classes_ = np.array([0, 1])
    upstream.predict.return_value = np.array([0, 1])
    constructor = Mock(return_value=upstream)
    name = "TabPFNClassifier" if classifier else "TabPFNRegressor"
    monkeypatch.setattr(f"tabpfn.{name}", constructor)
    monkeypatch.setattr("featune.tabpfn._checkpoint", lambda kind, version: path)
    X = pd.DataFrame({"x": [1.0, np.nan], "kind": ["a", None]})
    y = np.array([0, 1])
    cls = TabPFNClassifier if classifier else TabPFNRegressor
    model = cls(model_version=model_version, device="cpu", n_estimators=2, random_state=7).fit(X, y)
    assert model.provenance()["model"] == f"TabPFN-{model_version[1:]}"
    assert constructor.call_args.kwargs == {
        "model_path": path,
        "n_estimators": 2,
        "categorical_features_indices": [1],
        "device": "cpu",
        "random_state": 7,
        "n_preprocessing_jobs": 1,
        "fit_mode": "fit_preprocessors",
    }
    pd.testing.assert_frame_equal(upstream.fit.call_args.args[0], X)
    np.testing.assert_array_equal(model.predict(X), [0, 1])
    model.model_provenance_["model"] = "obsolete-preset"
    with pytest.raises(RuntimeError, match="different preset"):
        model.predict(X)


def test_safetensors_cache_symlink_preserves_format(tmp_path, monkeypatch):
    """Preserve safetensors detection after upstream resolves Hugging Face symlinks.

    Args:
        tmp_path (pathlib.Path): Isolated simulated cache with an extensionless blob.
        monkeypatch (pytest.MonkeyPatch): Replaces the cache lookup and expected digest.

    Returns:
        None: Returned paths resolve with the format suffix and share the verified inode.
    """
    import hashlib

    blob = tmp_path / "blobhash"
    blob.write_bytes(b"verified safetensors test payload")
    link = tmp_path / V35_WEIGHTS["classifier"]["filename"]
    link.symlink_to(blob)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda **kwargs: str(link))
    monkeypatch.setitem(V35_WEIGHTS["classifier"], "sha256", hashlib.sha256(blob.read_bytes()).hexdigest())
    from pathlib import Path

    for kind in ["classifier", "regressor"]:
        result = Path(_checkpoint(kind, "v3.5"))
        assert result.resolve().suffix == ".safetensors"
        assert result.samefile(blob)
