# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Pinned, DataFrame-native TabPFN estimators and portable fitted state.

Keeps foundation weights outside study artifacts. Downloads are restricted to
immutable licensed revisions and verified before use; inference remains local.

Created:
    2026-09-22
"""

import hashlib
import io
import json
import tempfile
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, is_classifier
from sklearn.utils.validation import check_is_fitted

TABPFN_VERSION = "9.0.0"
# Immutable upstream revisions and LFS content hashes, not the moving default model.
V35_WEIGHTS = dict.fromkeys(
    ("classifier", "regressor"),
    {
        "repo_id": "Prior-Labs/tabpfn_3_5",
        "revision": "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e",
        "filename": "tabpfn-v3.5-20260909.safetensors",
        "sha256": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
    },
)


WEIGHTS = {
    "classifier": {
        "repo_id": "Prior-Labs/TabPFN-v2-clf",
        "revision": "f851f2a3c941544733b712d8c0f96dfae9b28862",
        "filename": "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
        "sha256": "cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49",
    },
    "regressor": {
        "repo_id": "Prior-Labs/TabPFN-v2-reg",
        "revision": "4972a65a1b30806315c6f92499959ffbfc69a673",
        "filename": "tabpfn-v2-regressor.ckpt",
        "sha256": "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736",
    },
}
PRESETS = {"v2": WEIGHTS, "v3.5": V35_WEIGHTS}


def _checkpoint(kind, model_version="v2"):
    """Resolve and verify an immutable checkpoint in the Hugging Face cache.

    Args:
        kind (str): classifier or regressor.
        model_version (str): Explicit v2 (default) or v3.5 preset.

    Returns:
        str: Local checkpoint path with the expected SHA-256 digest.

    Raises:
        RuntimeError: Installed TabPFN version or checkpoint digest differs from the preset.
        OSError: Weights cannot be read; upstream download/access exceptions also propagate.

    Notes:
        Uses cached files before network access. HF_HUB_OFFLINE=1 forbids downloads.
        First uncached v3.5 use requires upstream license acceptance before download.
        No training rows are transmitted by checkpoint resolution.
    """
    from huggingface_hub import constants, hf_hub_download, try_to_load_from_cache

    if version("tabpfn") != TABPFN_VERSION:
        raise RuntimeError(f"This preset requires tabpfn=={TABPFN_VERSION}")
    spec = PRESETS[model_version][kind]
    options = {key: spec[key] for key in ("repo_id", "revision", "filename")}
    cached = try_to_load_from_cache(**options)
    if isinstance(cached, str):
        path = cached
    else:
        # Offline cache misses must not open a browser or contact the license server.
        # Acceptance belongs to the user; delegate it to the pinned upstream client.
        if model_version == "v3.5" and not constants.HF_HUB_OFFLINE:
            from tabpfn.browser_auth import ensure_license_accepted

            ensure_license_accepted(hf_repo_id="tabpfn_3_5")
        path = hf_hub_download(**options, local_files_only=constants.HF_HUB_OFFLINE)
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != spec["sha256"]:
        raise RuntimeError(f"TabPFN checkpoint checksum mismatch: {spec['filename']}")
    if Path(path).suffix == ".safetensors" and Path(path).resolve().suffix != ".safetensors":
        # Upstream resolves HF symlinks before detecting the format. A hard link
        # preserves the suffix without duplicating the 876 MB payload on disk.
        target = Path(path).parent / "featune" / spec["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.hardlink_to(Path(path).resolve())
        except FileExistsError:
            if not target.samefile(path):
                raise RuntimeError("Checkpoint format link points to a different file") from None
        path = str(target)
    return path


class _TabPFNBase(BaseEstimator):
    """Adapt selectable fixed TabPFN weights to Featune's cloneable DataFrame pipeline.

    Attributes:
        accepts_dataframe (bool): Bypasses generic scaling, imputation and one-hot encoding.
        preserve_missing_categories (bool): Keeps nulls in compiled categorical columns.
        model_ (sklearn estimator): Fitted upstream estimator, loaded lazily after deserialization.
        feature_names_in_ (numpy.ndarray): Ordered training feature names.
        n_features_in_ (int): Number of training columns.
        model_provenance_ (dict): Checkpoint identity used for this fitted state.
        classes_ (numpy.ndarray): Sorted class labels, classifiers only.

    Notes:
        Defaults to v2: 10,000 rows, 500 columns and 10 classes; CPU allows
        1,000 rows. Opt-in v3.5 allows 1,000,000 rows, 20,000 columns, 5,000 CPU rows.
        Artifacts contain fitted training context and must be
        treated as training data. Foundation weights are resolved separately by hash.
    """

    accepts_dataframe = True
    preserve_missing_categories = True

    def __init__(self, *, model_version="v2", n_estimators=4, device="auto", random_state=42):
        """Store cloneable inference settings without downloading or loading weights.

        Args:
            model_version (str): v2 by default; v3.5 requires its separate upstream license.
            n_estimators (int): Positive ensemble size; fixed at four by default.
            device (str): auto, cpu, cuda or a supported explicit PyTorch device.
            random_state (int): Seed for upstream preprocessing and ensemble generation.
        """
        self.model_version = model_version
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state

    def provenance(self):
        """Describe the immutable model identity without resolving local paths.

        Returns:
            dict: Package/model versions, upstream revision and weight checksum.
        """
        kind = "classifier" if is_classifier(self) else "regressor"
        if self.model_version not in PRESETS:
            raise ValueError("model_version must be 'v2' or 'v3.5'")
        return {
            "package": TABPFN_VERSION,
            "model": f"TabPFN-{self.model_version[1:]}",
            **PRESETS[self.model_version][kind],
        }

    def fit(self, X, y):
        """Fit preprocessing and training context using local fixed-weight inference.

        Args:
            X (pandas.DataFrame): Numeric/categorical columns with missing values allowed.
            y (array-like): Aligned target vector of shape (n_rows,).

        Returns:
            _TabPFNBase: This estimator with fitted state replaced.

        Raises:
            ValueError: Inputs, ensemble size or checkpoint capacity are invalid.
            RuntimeError: Weight checksum or package version is incompatible.

        Notes:
            Each call creates independent training context. No gradient training or
            automatic row/column subsampling is performed. Upstream device/fit errors
            propagate; there is no fallback to another estimator or a remote API.
        """
        from tabpfn import TabPFNClassifier as UpstreamClassifier
        from tabpfn import TabPFNRegressor as UpstreamRegressor
        from tabpfn.utils import infer_devices

        provenance = self.provenance()
        X = pd.DataFrame(X)
        y = np.asarray(y)
        if not X.columns.is_unique or X.empty or y.ndim != 1 or len(X) != len(y):
            raise ValueError("TabPFN requires unique nonempty columns and aligned one-dimensional targets")
        rows, columns, cpu_rows = (
            (10_000, 500, 1000) if self.model_version == "v2" else (1_000_000, 20_000, 5000)
        )
        if len(X) > rows or X.shape[1] > columns:
            raise ValueError(
                f"Pinned {provenance['model']} supports at most {rows:,} training rows and {columns:,} columns"
            )
        if not isinstance(self.n_estimators, int) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if len(X) > cpu_rows and all(d.type == "cpu" for d in infer_devices(self.device)):
            raise ValueError(
                f"Pinned {provenance['model']} CPU inference supports at most {cpu_rows:,} training rows"
            )
        classification = is_classifier(self)
        if classification and not 2 <= len(np.unique(y)) <= 10:
            raise ValueError("Pinned TabPFN classification requires 2 to 10 classes")
        kind = "classifier" if classification else "regressor"
        cls = UpstreamClassifier if classification else UpstreamRegressor
        categorical = [i for i, dtype in enumerate(X.dtypes) if not pd.api.types.is_numeric_dtype(dtype)]
        model = cls(
            model_path=_checkpoint(kind, self.model_version),
            n_estimators=self.n_estimators,
            categorical_features_indices=categorical,
            device=self.device,
            random_state=self.random_state,
            n_preprocessing_jobs=1,
            fit_mode="fit_preprocessors",
        )
        model.fit(X, y)
        self.model_ = model
        self.model_provenance_ = self.provenance()
        self.__dict__.pop("_fit_archive", None)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        if classification:
            self.classes_ = model.classes_
        return self

    def _prediction_input(self, X):
        """Validate inference columns and restore a serialized upstream fit if needed.

        Args:
            X (pandas.DataFrame): Rows with training columns in their original order.

        Returns:
            pandas.DataFrame: Validated inputs with missing values retained.

        Raises:
            NotFittedError: No training context exists.
            ValueError: Column identities/order differ from training.
        """
        check_is_fitted(self, "feature_names_in_")
        if self.model_provenance_ != self.provenance():
            raise RuntimeError("Fitted TabPFN state belongs to a different preset; refit explicitly")
        X = pd.DataFrame(X)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Prediction columns must match training names and order")
        if "model_" not in self.__dict__:
            from tabpfn.model_loading import load_fitted_tabpfn_model

            kind = "classifier" if is_classifier(self) else "regressor"
            checkpoint = _checkpoint(kind, self.model_version)
            # Replace the originating machine's cache path before upstream restoration.
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "state.tabpfn_fit"
                with zipfile.ZipFile(io.BytesIO(self._fit_archive)) as source:
                    with zipfile.ZipFile(path, "w") as target:
                        for name in source.namelist():
                            data = source.read(name)
                            if name == "init_params.json":
                                params = json.loads(data)
                                params["model_path"] = checkpoint
                                data = json.dumps(params).encode()
                            target.writestr(name, data)
                self.model_ = load_fitted_tabpfn_model(path, device=self.device)
            del self._fit_archive
        return X

    def predict(self, X):
        """Predict class labels or regression targets using the fitted training context.

        Args:
            X (pandas.DataFrame): Inference rows with unchanged column names/order.

        Returns:
            numpy.ndarray: Predictions of shape (n_rows,) in original target units/labels.
        """
        X = self._prediction_input(X)
        return self.model_.predict(X)

    def __getstate__(self):
        """Serialize fitted state without foundation weights or live device allocations.

        Returns:
            dict: Cloneable settings plus a CPU-portable upstream fitted-state archive.

        Notes:
            Restoring prediction needs the pinned weights in cache or download access.
            The archive includes training context; only load trusted joblib artifacts.
        """
        state = self.__dict__.copy()
        model = state.pop("model_", None)
        if model is not None:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "state.tabpfn_fit"
                model.save_fit_state(path)
                state["_fit_archive"] = path.read_bytes()
        return state


class TabPFNClassifier(ClassifierMixin, _TabPFNBase):
    """Classify with a pinned TabPFN checkpoint and native categorical preprocessing.

    Notes:
        Inherits configuration and fitted attributes from _TabPFNBase. Class probability
        columns follow classes_; labels may be strings or integers.
    """

    def predict_proba(self, X):
        """Estimate probabilities in fitted class order.

        Args:
            X (pandas.DataFrame): Inference rows with unchanged column names/order.

        Returns:
            numpy.ndarray: Probabilities of shape (n_rows, n_classes).
        """
        X = self._prediction_input(X)
        return self.model_.predict_proba(X)


class TabPFNRegressor(RegressorMixin, _TabPFNBase):
    """Regress with pinned TabPFN weights and training-only target preprocessing.

    Notes:
        Inherits configuration, prediction and fitted attributes from _TabPFNBase.
        Outputs retain the original target units.
    """


def default_estimator(metric="auc", random_state=42):
    """Choose the pinned classifier or regressor from an explicit metric family.

    Args:
        metric (str): Supported classification or regression metric name.
        random_state (int): Seed shared with study/evaluator defaults.

    Returns:
        TabPFNClassifier or TabPFNRegressor: Unfitted local estimator; no weight download yet.

    Raises:
        ValueError: The metric does not identify a supported task.
    """
    if metric in {"auc", "auc_ovr", "accuracy", "f1", "log_loss"}:
        return TabPFNClassifier(random_state=random_state)
    if metric in {"rmse", "mae", "r2"}:
        return TabPFNRegressor(random_state=random_state)
    raise ValueError(f"Cannot select a default estimator for metric: {metric}")
