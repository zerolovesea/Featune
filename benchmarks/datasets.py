# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Public benchmark data and semantic ablations.

Loads cached public tasks and semantic ablations. Task construction avoids
learning target preprocessing from the full pre-split dataset.

Created:
    2026-09-22
"""

import zipfile
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    fetch_covtype,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_wine,
)

from featune import DatasetSchema, FieldSchema

SOURCES = {
    "adult": "https://archive.ics.uci.edu/dataset/2/adult",
    "california": "https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset",
    "covtype": "https://archive.ics.uci.edu/dataset/31/covertype",
    "online_retail": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
    "cancer": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
    "diabetes": "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html",
    "wine": "scikit-learn bundled wine recognition dataset",
    "digits": "scikit-learn bundled handwritten digits dataset",
}
CLASSIFICATION = {"adult", "cancer", "covtype", "wine", "digits"}


def load_dataset(name, anonymous=False, shuffled=False, seed=42, data_dir="runs/datasets"):
    """Load a public task and optionally anonymize or shuffle its semantic schema.

    Args:
        name (str): Dataset key registered in SOURCES.
        anonymous (bool): Replace names/descriptions/objective with semantic-blinding placeholders.
        shuffled (bool): Apply a seeded description derangement while preserving data and field names.
        seed (int): Seed for deterministic random choices.
        data_dir (str or Path): Local public-dataset download/cache directory.

    Returns:
        tuple[pandas.DataFrame, pandas.Series, DatasetSchema]: Aligned features, targets and field
        semantics.

    Raises:
        ValueError: Dataset name is not registered.

    Notes:
        Downloads missing public archives into data_dir; cached files are reused.
        Online Retail keeps positive-price/positive-quantity rows and preserves raw
        Quantity targets without learned clipping. Ablations alter only schema/name
        semantics, preserving row values/order and target alignment. Network, archive
        and optional Excel-reader failures propagate.
    """
    if name not in SOURCES:
        raise ValueError(f"Unknown dataset: {name}")
    cache = Path(data_dir)
    cache.mkdir(parents=True, exist_ok=True)
    descriptions = {}
    if name == "adult":
        archive = cache / "adult.zip"
        if not archive.exists():
            response = httpx.get(
                "https://archive.ics.uci.edu/static/public/2/adult.zip", timeout=60, follow_redirects=True
            )
            response.raise_for_status()
            temporary = archive.with_suffix(".tmp")
            temporary.write_bytes(response.content)
            temporary.replace(archive)
        columns = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education_num",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "native_country",
            "income",
        ]
        with zipfile.ZipFile(archive) as files:
            frames = [
                pd.read_csv(
                    files.open(filename), names=columns, skipinitialspace=True, na_values="?", comment="|"
                )
                for filename in ["adult.data", "adult.test"]
            ]
        frame = pd.concat(frames, ignore_index=True)
        y = frame.pop("income").str.rstrip(".").eq(">50K").astype(int)
        X = frame
        descriptions = {
            "age": "Age in years",
            "workclass": "Employment ownership and work class",
            "fnlwgt": "Census final sampling weight; not an income measurement",
            "education": "Highest education qualification",
            "education_num": "Numeric educational attainment level",
            "marital_status": "Marital status category",
            "occupation": "Occupation category",
            "relationship": "Relationship to household reference person",
            "race": "Recorded census race category",
            "sex": "Recorded census sex category",
            "capital_gain": "Annual capital gains in dollars",
            "capital_loss": "Annual capital losses in dollars",
            "hours_per_week": "Typical working hours per week",
            "native_country": "Country of origin",
        }
        objective = "Predict whether annual income exceeds 50000 dollars from census attributes"
    elif name == "california":
        data = fetch_california_housing(as_frame=True, data_home=cache)
        X, y = data.data.copy(), data.target.copy()
        descriptions = {
            "MedInc": "Median income in a census block group, tens of thousands of dollars",
            "HouseAge": "Median house age in years",
            "AveRooms": "Average rooms per household",
            "AveBedrms": "Average bedrooms per household",
            "Population": "Block group population",
            "AveOccup": "Average household occupancy",
            "Latitude": "Block group latitude",
            "Longitude": "Block group longitude",
        }
        objective = "Predict median house value in units of 100000 dollars"
    elif name == "covtype":
        data = fetch_covtype(as_frame=True, data_home=cache)
        X, y = data.data.copy(), data.target.copy()
        descriptions = {
            name: (
                "Cartographic and environmental measurement used to classify forest cover type "
                f"({name.replace('_', ' ')})"
            )
            for name in X
        }
        objective = "Predict one of seven forest cover types from cartographic and environmental measurements"
    elif name == "online_retail":
        # The UCI workbook has two sheets and is converted to a transaction-level
        # demand task. Quantity is the target; invoice identifiers and descriptions
        # are excluded to avoid direct target leakage and high-cardinality text.
        import io

        archive = cache / "online-retail-ii.zip"
        if not archive.exists():
            response = httpx.get(
                "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
                timeout=180,
                follow_redirects=True,
            )
            response.raise_for_status()
            temporary = archive.with_suffix(".tmp")
            temporary.write_bytes(response.content)
            temporary.replace(archive)
        with zipfile.ZipFile(archive) as files:
            workbook = files.read("online_retail_II.xlsx")
        frame = pd.concat(
            pd.read_excel(io.BytesIO(workbook), sheet_name=sheet)
            for sheet in ["Year 2009-2010", "Year 2010-2011"]
        )
        frame = frame.dropna(subset=["Quantity", "Price", "InvoiceDate", "StockCode", "Country"])
        frame = frame[(frame["Quantity"] > 0) & (frame["Price"] > 0)].copy()
        frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
        frame = frame.dropna(subset=["InvoiceDate"])
        frame["invoice_month"] = frame["InvoiceDate"].dt.month.astype(float)
        frame["invoice_dayofweek"] = frame["InvoiceDate"].dt.dayofweek.astype(float)
        frame["invoice_hour"] = frame["InvoiceDate"].dt.hour.astype(float)
        X = frame[
            ["StockCode", "Price", "Country", "invoice_month", "invoice_dayofweek", "invoice_hour"]
        ].rename(columns={"StockCode": "stock_code", "Price": "unit_price", "Country": "country"})
        # Keep the original target: a full-data quantile would leak held-out labels.
        y = frame["Quantity"].astype(float)
        descriptions = {
            "stock_code": "Product stock keeping unit for the retail line",
            "unit_price": "Unit selling price in pounds sterling",
            "country": "Customer country",
            "invoice_month": "Calendar month of the invoice",
            "invoice_dayofweek": "Day of week when the invoice was issued",
            "invoice_hour": "Hour of day when the invoice was issued",
        }
        objective = (
            "Predict purchased quantity for a retail transaction from product, price, geography and time"
        )
    elif name in {"wine", "digits"}:
        data = load_wine(as_frame=True) if name == "wine" else load_digits(as_frame=True)
        X, y = data.data.copy(), data.target.copy()
        descriptions = {
            field: (
                f"Wine chemistry measurement: {field.replace('_', ' ')}"
                if name == "wine"
                else f"Handwritten digit pixel intensity: {field.replace('_', ' ')}"
            )
            for field in X
        }
        objective = (
            "Classify wine cultivar from chemistry" if name == "wine" else "Classify handwritten digit"
        )
    else:
        data = (
            load_breast_cancer(as_frame=True)
            if name == "cancer"
            else load_diabetes(as_frame=True, scaled=False)
        )
        X, y = data.data.copy(), data.target.copy()
        descriptions = {
            "age": "Age in years",
            "sex": "Recorded sex code",
            "bmi": "Body mass index",
            "bp": "Average blood pressure",
            "s1": "Total serum cholesterol",
            "s2": "Low-density lipoproteins",
            "s3": "High-density lipoproteins",
            "s4": "Total cholesterol divided by HDL",
            "s5": "Log-transformed serum triglycerides",
            "s6": "Blood glucose level",
        }
        objective = (
            "Predict benign vs malignant breast mass"
            if name == "cancer"
            else "Predict one-year disease progression"
        )
    fields, names = [], {}
    for index, original in enumerate(X):
        name_out = f"x{index}" if anonymous else original.replace(" ", "_")
        names[original] = name_out
        description = (
            "Measurement; semantics withheld"
            if anonymous
            else descriptions.get(original, f"{original} of cell nuclei in digitized breast mass images")
        )
        dtype = "numeric" if pd.api.types.is_numeric_dtype(X[original]) else "categorical"
        if dtype == "categorical":
            X[original] = X[original].astype("category")
        fields.append(FieldSchema(name=name_out, dtype=dtype, description=description))
    if shuffled:
        # A seeded derangement mismatches every description while preserving field names and types.
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(fields))
        shifted = np.roll(order, 1)
        replacements = {int(a): fields[int(b)].description for a, b in zip(order, shifted)}
        fields = [
            field.model_copy(update={"description": replacements[index]})
            for index, field in enumerate(fields)
        ]
    if anonymous:
        objective = "Improve held-out predictive accuracy"
    return X.rename(columns=names), y, DatasetSchema(fields=fields, objective=objective)
