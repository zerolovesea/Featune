# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Alternative sklearn estimators with learned tabular embeddings.

Provides sklearn-compatible neural estimators with training-only encoders.
PyTorch is included in the default installation; this module opts into gradient training.

Created:
    2026-09-21
"""

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn


class _TabularNet(nn.Module):
    """Combine continuous values and learned category embeddings in a feed-forward network.

    Attributes:
        embeddings (nn.ModuleList): One embedding table per categorical field.
        network (nn.Sequential): Hidden linear/ReLU/dropout blocks and final prediction layer.

    Notes:
        Inputs must already be encoded with training-only vocabularies and numeric scaling.
    """

    def __init__(self, n_numeric, cardinalities, embedding_dim, hidden_dims, dropout, outputs):
        """Create embedding tables and hidden/output layers for the tabular model.

        Args:
            n_numeric (int): Number of continuous input columns.
            cardinalities (sequence[int]): Embedding table sizes, including the unknown-category slot.
            embedding_dim (int): Positive upper bound for each categorical embedding width.
            hidden_dims (sequence[int]): Positive hidden-layer widths; length controls network depth.
            dropout (float): Hidden-layer dropout probability in [0, 1).
            outputs (int): Output width: class count for classification or one for regression.
        """
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, min(embedding_dim, max(2, size // 2))) for size in cardinalities]
        )
        width = n_numeric + sum(layer.embedding_dim for layer in self.embeddings)
        layers = []
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)])
            width = hidden
        self.network = nn.Sequential(*layers, nn.Linear(width, outputs))

    def forward(self, numeric, categorical):
        """Concatenate numeric values and categorical embeddings before prediction.

        Args:
            numeric (torch.Tensor): Float tensor shaped (n_rows, n_numeric) on the model device.
            categorical (torch.Tensor): Integer category IDs shaped (n_rows, n_categorical).

        Returns:
            torch.Tensor: Raw outputs shaped (n_rows, outputs); no softmax or inverse target scaling.
        """
        values = [numeric] + [
            embedding(categorical[:, index]) for index, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(values, dim=1))


class _TorchBase(BaseEstimator):
    """Provide cloneable sklearn-style fitting for alternative tabular PyTorch estimators.

    Attributes:
        accepts_dataframe (bool): Signals build_pipeline to bypass generic one-hot preprocessing.
        feature_names_in_ (numpy.ndarray): Fitted DataFrame columns in required inference order.
        n_features_in_ (int): Total fitted input-column count.
        numeric_columns_ (list): Columns detected as numeric during fit.
        categorical_columns_ (list): Remaining columns represented by embeddings.
        categories_ (dict): Per-column string-to-ID maps; zero is reserved for unknown categories.
        center_ (numpy.ndarray): Training numeric medians for imputation and centering.
        scale_ (numpy.ndarray): Training numeric standard deviations with a 1e-6 floor.
        model_ (nn.Module): Fitted network on the selected device.
        loss_curve_ (list[float]): Mean training loss per epoch.
        classes_ (numpy.ndarray): Sorted classification labels, classifier only.
        target_center_ (float): Training target mean, regressor only.
        target_scale_ (float): Floored training target standard deviation, regressor only.

    Notes:
        Hyperparameters are stored unchanged for sklearn.clone; see __init__.
        CPU seed behavior is deterministic for supported operations; accelerator
        determinism depends on the installed PyTorch backend and hardware.
    """

    accepts_dataframe = True

    def __init__(
        self,
        hidden_dims=(64, 32),
        learning_rate=0.001,
        embedding_dim=8,
        dropout=0.1,
        epochs=30,
        batch_size=128,
        weight_decay=0.0001,
        random_state=42,
        device="cpu",
    ):
        """Store network, optimizer, batching and execution settings without allocating a model.

        Args:
            hidden_dims (sequence[int]): Positive hidden-layer widths; length controls network depth.
            learning_rate (float): Positive AdamW learning rate.
            embedding_dim (int): Positive upper bound for each categorical embedding width.
            dropout (float): Hidden-layer dropout probability in [0, 1).
            epochs (int): Positive number of complete training passes.
            batch_size (int): Positive row count per training/inference batch.
            weight_decay (float): AdamW weight-decay coefficient.
            random_state (int): Seed for model initialization, dropout and minibatch permutations.
            device (str or torch.device): Execution device, such as cpu or cuda:0.
        """
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device

    def _encode(self, X, fitting=False):
        """Encode a DataFrame using fitted numeric and categorical training state.

        Args:
            X (pandas.DataFrame): Input feature rows with unique columns and schema-compatible types.
            fitting (bool): Learn training-only state when True; otherwise reuse previously fitted state.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: CPU float32 numeric and int64 categorical matrices.

        Raises:
            ValueError: Inference columns do not match fitted names and order.

        Notes:
            With fitting=True, replaces vocabularies, medians and scales. Unknown categories
            map to zero; non-finite numeric values are imputed then scaled/clipped.
            Standalone categorical inputs use string conversion, including missing-as-empty.
        """
        X = pd.DataFrame(X)
        if fitting:
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            self.n_features_in_ = X.shape[1]
            self.numeric_columns_ = list(X.select_dtypes(include="number").columns)
            self.categorical_columns_ = [name for name in X if name not in self.numeric_columns_]
            self.categories_ = {
                name: {
                    value: index + 1
                    for index, value in enumerate(sorted(X[name].fillna("").astype(str).unique()))
                }
                for name in self.categorical_columns_
            }
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Inference columns must match fitted columns in order")
        numeric = X[self.numeric_columns_].to_numpy(dtype=np.float32, copy=True)
        numeric[~np.isfinite(numeric)] = np.nan
        if fitting:
            self.center_ = np.asarray(
                [np.nanmedian(column) if np.isfinite(column).any() else 0.0 for column in numeric.T],
                dtype=np.float32,
            )
        numeric = np.where(np.isnan(numeric), self.center_, numeric)
        if fitting:
            self.scale_ = np.maximum(numeric.std(axis=0), 1e-6)
        numeric = np.clip((numeric - self.center_) / self.scale_, -1e6, 1e6)
        categorical = (
            np.column_stack(
                [
                    X[name]
                    .fillna("")
                    .astype(str)
                    .map(self.categories_[name])
                    .fillna(0)
                    .to_numpy(dtype=np.int64)
                    for name in self.categorical_columns_
                ]
            )
            if self.categorical_columns_
            else np.empty((len(X), 0), dtype=np.int64)
        )
        return torch.from_numpy(numeric), torch.from_numpy(categorical)

    def fit(self, X, y):
        """Train a classifier or regressor with fold-local encoding and AdamW.

        Args:
            X (pandas.DataFrame): Input feature rows with unique columns and schema-compatible types.
            y (array-like): One-dimensional targets aligned with X; never sent to the LLM.

        Returns:
            _TorchBase: This fitted estimator with model_ and loss_curve_.

        Raises:
            ValueError: Network limits/targets are invalid, classification has fewer than two classes, or
                loss is non-finite.

        Notes:
            Replaces prior fitted state. The torch RNG fork restores caller RNG state after
            training; a separate seeded generator controls minibatch order. Regressors learn
            target centering/scaling only from supplied training targets.
        """
        if (
            self.epochs < 1
            or self.batch_size < 1
            or self.learning_rate <= 0
            or self.embedding_dim < 1
            or any(width < 1 for width in self.hidden_dims)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("Invalid neural network hyperparameters")
        numeric, categorical = self._encode(X, fitting=True)
        classification = isinstance(self, ClassifierMixin)
        if classification:
            self.classes_, target = np.unique(y, return_inverse=True)
            if len(self.classes_) < 2:
                raise ValueError("Classification requires at least two classes")
            target = torch.tensor(target, dtype=torch.long)
            outputs, criterion = len(self.classes_), nn.CrossEntropyLoss()
        else:
            values = np.asarray(y, dtype=np.float32)
            # Learn target scaling from this fit only; CV supplies training-fold targets here.
            self.target_center_, self.target_scale_ = float(values.mean()), max(float(values.std()), 1e-6)
            target = torch.tensor((values - self.target_center_) / self.target_scale_).reshape(-1, 1)
            outputs, criterion = 1, nn.MSELoss()
        if len(target) != len(numeric) or not torch.isfinite(target).all():
            raise ValueError("Invalid targets")
        devices = [torch.device(self.device).index or 0] if str(self.device).startswith("cuda") else []
        # Restore caller RNG state after seeded initialization, dropout and training.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.random_state)
            self.model_ = _TabularNet(
                len(self.numeric_columns_),
                [len(self.categories_[name]) + 1 for name in self.categorical_columns_],
                self.embedding_dim,
                self.hidden_dims,
                self.dropout,
                outputs,
            ).to(self.device)
            optimizer = torch.optim.AdamW(
                self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
            generator = torch.Generator().manual_seed(self.random_state)
            self.loss_curve_ = []
            self.model_.train()
            for _ in range(self.epochs):
                order = torch.randperm(len(target), generator=generator)
                total_loss = 0.0
                for indexes in order.split(self.batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    prediction = self.model_(
                        numeric[indexes].to(self.device), categorical[indexes].to(self.device)
                    )
                    loss = criterion(prediction, target[indexes].to(self.device))
                    if not torch.isfinite(loss):
                        raise ValueError("Neural training produced a non-finite loss")
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(indexes)
                self.loss_curve_.append(total_loss / len(target))
        self.model_.eval()
        return self

    def _predict_raw(self, X):
        """Run batched inference without gradient recording.

        Args:
            X (pandas.DataFrame): Input feature rows with unique columns and schema-compatible types.

        Returns:
            torch.Tensor: Concatenated CPU raw outputs shaped (n_rows, n_outputs).

        Raises:
            NotFittedError: No trained model_ is available.
            ValueError: Column identities/order differ from training.

        Notes:
            X must contain at least one row. The network is placed in evaluation mode by fit.
        """
        check_is_fitted(self, "model_")
        numeric, categorical = self._encode(X)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(numeric), self.batch_size):
                outputs.append(
                    self.model_(
                        numeric[start : start + self.batch_size].to(self.device),
                        categorical[start : start + self.batch_size].to(self.device),
                    ).cpu()
                )
        return torch.cat(outputs)


class TorchClassifier(ClassifierMixin, _TorchBase):
    """Expose the tabular network through sklearn classifier prediction methods.

    Notes:
        Inherits constructor settings and fitted encoding attributes from _TorchBase.
        classes_ defines the probability-column and returned-label order.
    """

    def predict_proba(self, X):
        """Convert raw class logits into normalized probabilities.

        Args:
            X (pandas.DataFrame): Input feature rows with unique columns and schema-compatible types.

        Returns:
            numpy.ndarray: Float probabilities shaped (n_rows, n_classes), ordered by classes_.

        Raises:
            NotFittedError: The classifier has not been fitted.
        """
        return self._predict_raw(X).softmax(dim=1).numpy()

    def predict(self, X):
        """Choose the label with maximum predicted class probability.

        Args:
            X (pandas.DataFrame): Input feature rows with unique columns and schema-compatible types.

        Returns:
            numpy.ndarray: One original-domain class label per input row.

        Raises:
            NotFittedError: The classifier has not been fitted.
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TorchRegressor(RegressorMixin, _TorchBase):
    """Expose the tabular network through sklearn regression prediction methods.

    Notes:
        Inherits _TorchBase settings and training-only encoding state. Predictions are
        returned in the original target units using fitted target_center_/target_scale_.
    """

    def predict(self, X):
        """Undo training-target scaling for raw regression predictions.

        Args:
            X (pandas.DataFrame): Input feature rows with unique columns and schema-compatible types.

        Returns:
            numpy.ndarray: One floating target estimate per input row in original units.

        Raises:
            NotFittedError: The regressor has not been fitted.
        """
        return self._predict_raw(X).numpy().ravel() * self.target_scale_ + self.target_center_
