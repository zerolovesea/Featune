# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Feature-search example with the pinned TabPFN default.

Demonstrates feature search with a separate untouched test split.
First use downloads the pinned model weights; subsequent runs may run offline.
Running the example writes local pipeline, feature and trial artifacts.

Created:
    2026-09-21
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import featune


def make_dataset(seed=42, n=600):
    """Generate a reproducible synthetic repayment-burden classification task.

    Args:
        seed (int): Seed for deterministic random choices.
        n (int): Number of synthetic observations to generate.

    Returns:
        tuple[pandas.DataFrame, pandas.Series, DatasetSchema]: Features, aligned binary target and
        semantics.
    """
    rng = np.random.default_rng(seed)
    income = rng.lognormal(9, 0.6, n)
    loan = rng.lognormal(10, 0.9, n)
    age = rng.integers(20, 70, n)
    X = pd.DataFrame({"loan": loan, "income": income, "age": age})
    y = pd.Series((loan / income + rng.normal(0, 0.2, n) > 3).astype(int), name="default")
    schema = featune.DatasetSchema(
        fields=[
            featune.FieldSchema(
                name="loan",
                description="Outstanding loan principal at application",
                unit="CNY",
                partition="finance",
            ),
            featune.FieldSchema(
                name="income",
                description="Monthly disposable income at application",
                unit="CNY/month",
                partition="finance",
            ),
            featune.FieldSchema(
                name="age", description="Applicant age at application", unit="year", partition="profile"
            ),
        ],
        partitions={"finance": "Pre-decision financial information", "profile": "Applicant demographics"},
        objective="Predict future default using only information available at application",
    )
    return X, y, schema


def run(output="runs/quickstart"):
    """Execute a TabPFN beam-search example and export its selected artifacts.

    Args:
        output (str or Path): Output directory for generated experiment artifacts.

    Returns:
        tuple[FeatureStudy, float]: Completed study and untouched holdout ROC AUC.

    Notes:
        Creates output directories and writes pipeline, features and trial records.
        The fixed outer holdout is never supplied to optimize; existing output files
        with matching names are replaced.
    """
    X, y, schema = make_dataset()
    train, test = train_test_split(np.arange(len(X)), test_size=0.25, stratify=y, random_state=42)
    study = featune.create_study(
        metric="auc", sampler=featune.RandomSampler(seed=42), search_strategy="beam", beam_width=3
    )
    study.optimize(
        X.iloc[train],
        y.iloc[train],
        schema,
        evaluator=featune.CVEvaluator(cv=3),
        n_trials=3,
    )
    test_auc = roc_auc_score(y.iloc[test], study.export_pipeline().predict_proba(X.iloc[test])[:, 1])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    study.export_pipeline(output / "pipeline.joblib")
    study.export_features(output / "features.json")
    study.trials_dataframe().to_json(output / "trials.json", orient="records", indent=2)
    print(
        f"Baseline CV AUC: {study.baseline_trial.value:.6f}; best CV AUC: {study.best_value:.6f}; test AUC: {test_auc:.6f}"
    )
    return study, test_auc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    run()
