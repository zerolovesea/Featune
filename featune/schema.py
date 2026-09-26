# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Semantic input contracts and the closed feature language.

Defines the trust boundary for metadata and the closed DSL. Validation never
executes generated Python, SQL or arbitrary expressions.

Created:
    2026-09-21
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    """Base model for immutable, finite, closed-schema API contracts.

    Notes:
        Unknown keys and non-finite numbers are rejected. Frozen models prevent attribute
        rebinding, but contained lists and dictionaries are not deeply immutable.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        revalidate_instances="always",
    )


class FieldSchema(Contract):
    """Describe one raw field and its availability to the feature search.

    Attributes:
        name (str): Unique raw DataFrame column name.
        dtype (str): numeric, categorical or datetime.
        description (str): Business meaning; treated as untrusted prompt data.
        partition (str or None): Key into DatasetSchema.partitions, not a CV split.
        unit (str or None): Descriptive measurement unit; no dimensional algebra is enforced.
        encoding (str or None): Meaning of category codes, sent as bounded prompt metadata.
        available_at (str or None): Business-time availability, not automatically verified.
        entity (str or None): Entity label used as a fallback retrieval concept.
        semantic_tags (list[str]): Up to 20 short semantic labels; the first defines the preferred
            concept.
        exclude (bool): Exclude the field from compilation, retrieval and model inputs.
    """

    name: str = Field(min_length=1, max_length=200)
    dtype: Literal["numeric", "categorical", "datetime"] = "numeric"
    description: str = Field(min_length=1, max_length=2000)
    partition: str | None = None
    unit: str | None = None
    encoding: str | None = Field(default=None, max_length=500)
    available_at: str | None = Field(default=None, max_length=500)
    entity: str | None = Field(default=None, max_length=200)
    semantic_tags: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list, max_length=20
    )
    exclude: bool = False


class DatasetSchema(Contract):
    """Define the usable raw-field contract and target-free task context.

    Attributes:
        fields (list[FieldSchema]): Unique fields; at least one must be usable.
        partitions (dict[str, str]): Business-domain descriptions keyed by partition name.
        objective (str): Task description; must not disclose outer-test results.
        target_definition (str or None): Meaning, horizon and positive class of the target.
        prediction_point (str or None): When features must be available for prediction.
        search_guidance (str or None): Optional, untrusted hypotheses to explore with an LLM sampler.
    """

    fields: list[FieldSchema] = Field(min_length=1)
    partitions: dict[str, str] = Field(default_factory=dict)
    objective: str = Field(default="Improve out-of-sample predictive performance", max_length=4000)
    target_definition: str | None = Field(default=None, max_length=2000)
    prediction_point: str | None = Field(default=None, max_length=1000)
    search_guidance: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_fields(self):
        """Validate field uniqueness, partition references and usable capacity.

        Returns:
            DatasetSchema: This validated instance.

        Raises:
            ValueError: A field name is duplicated, a partition is unknown, or all fields are excluded.
        """
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate schema field names")
        for field in self.fields:
            if field.partition is not None and field.partition not in self.partitions:
                raise ValueError(f"Undefined partition: {field.partition}")
        if not any(not f.exclude for f in self.fields):
            raise ValueError("At least one usable field is required")
        return self

    @property
    def usable(self) -> dict[str, FieldSchema]:
        """Return included fields indexed by their raw column names.

        Returns:
            dict[str, FieldSchema]: A newly constructed mapping preserving schema order.
        """
        return {f.name: f for f in self.fields if not f.exclude}

    def validate_frame(self, X: pd.DataFrame):
        """Check input columns and declared numeric/datetime compatibility.

        Args:
            X (pandas.DataFrame): Input feature rows with unique columns and schema-compatible types.

        Returns:
            None: X is checked without mutation.

        Raises:
            ValueError: X is not a uniquely named DataFrame, required fields are absent, or types are
                incompatible.

        Notes:
            Extra columns are allowed but are not compiled. Categorical values are normalized
            by the compiler; this check does not infer business-time availability.
        """
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("X must be a DataFrame with unique string column names")
        missing = set(self.usable) - set(X.columns)
        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")
        for name, field in self.usable.items():
            if field.dtype == "numeric" and not pd.api.types.is_numeric_dtype(X[name]):
                raise ValueError(f"Expected numeric column: {name}")
            if field.dtype == "datetime":
                pd.to_datetime(X[name], errors="raise", utc=True)


# Arity and type signatures form the entire executable surface of the DSL.
OPS = {
    "add": ("numeric", "numeric"),
    "subtract": ("numeric", "numeric"),
    "multiply": ("numeric", "numeric"),
    "divide": ("numeric", "numeric"),
    "log1p_abs": ("numeric",),
    "sqrt_abs": ("numeric",),
    "square": ("numeric",),
    "abs": ("numeric",),
    "clip": ("numeric",),
    "is_missing": ("any",),
    "frequency": ("categorical",),
    "cross": ("categorical", "categorical"),
    "group_mean": ("categorical", "numeric"),
    "group_std": ("categorical", "numeric"),
    "year": ("datetime",),
    "month": ("datetime",),
    "dayofweek": ("datetime",),
    "days_between": ("datetime", "datetime"),
}
Operation = Literal[
    "add",
    "subtract",
    "multiply",
    "divide",
    "log1p_abs",
    "sqrt_abs",
    "square",
    "abs",
    "clip",
    "is_missing",
    "frequency",
    "cross",
    "group_mean",
    "group_std",
    "year",
    "month",
    "dayofweek",
    "days_between",
]


class Hypothesis(Contract):
    """Record a testable explanation without claiming causal identification.

    Attributes:
        concept (str): Short business or transformation concept.
        rationale (str): Reason the proposed expression might improve prediction.
        expected_direction (str): increasing, decreasing or unknown descriptive target association.
    """

    concept: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=1, max_length=2000)
    expected_direction: Literal["increasing", "decreasing", "unknown"] = "unknown"


class FeatureSpec(Contract):
    """Represent one expression in the closed, non-executable feature DSL.

    Attributes:
        name (str): ASCII output identifier, unique within the candidate/schema.
        op (str): Operator from the closed OPS registry.
        inputs (list[str]): One or two typed inputs, respecting operator arity.
        params (dict[str, float]): Empty except for clip, which requires lower and upper.
        hypothesis (Hypothesis): Testable semantic rationale and expected direction.
    """

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,99}$")
    op: Operation
    inputs: list[str] = Field(min_length=1, max_length=2)
    params: dict[str, Annotated[float, Field(strict=True)]] = Field(default_factory=dict)
    hypothesis: Hypothesis

    @model_validator(mode="after")
    def validate_operation(self):
        """Check operator arity and its exact allowed parameter set.

        Returns:
            FeatureSpec: This validated instance.

        Raises:
            ValueError: Arity or parameter keys are invalid, or clip bounds are not increasing.
        """
        if len(self.inputs) != len(OPS[self.op]):
            raise ValueError(f"Invalid arity for {self.op}")
        allowed = {"lower", "upper"} if self.op == "clip" else set()
        if set(self.params) != allowed:
            raise ValueError(f"{self.op} requires exactly these parameters: {sorted(allowed)}")
        if self.op == "clip" and self.params["lower"] >= self.params["upper"]:
            raise ValueError("clip lower must be less than upper")
        return self

    @property
    def signature(self) -> str:
        """Build an alias-independent signature for immediate input expressions.

        Returns:
            str: Stable JSON of operator, normalized input names and parameters.

        Notes:
            Commutative inputs are sorted. This signature does not recursively resolve
            derived aliases; FeatureSet.expression_hash supplies recursive identity.
        """
        inputs = sorted(self.inputs) if self.op in {"add", "multiply", "cross"} else self.inputs
        return json.dumps([self.op, inputs, self.params], sort_keys=True)


class Proposal(Contract):
    """Carry a bounded feature extension and approved model-parameter choices.

    Attributes:
        features (list[FeatureSpec]): Up to 32 new features in dependency order.
        model_params (dict): Estimator overrides checked against the study parameter space.
    """

    features: list[FeatureSpec] = Field(default_factory=list, max_length=32)
    remove: list[str] = Field(default_factory=list, max_length=64)
    model_params: dict[str, int | float | str | bool | list[int]] = Field(default_factory=dict)


def validate_features(schema: DatasetSchema, features: list[FeatureSpec], max_features: int = 64):
    """Validate dependency order, names, types and expression uniqueness.

    Args:
        schema (DatasetSchema): Validated field types, semantics, exclusions and task objective.
        features (FeatureSet or sequence[FeatureSpec]): Controlled derived features, excluding raw
            columns.
        max_features (int): Maximum derived-feature count; at most 64 in study/DSL validation.

    Returns:
        dict[str, str]: Raw and derived column names mapped to output dtypes.

    Raises:
        ValueError: The feature limit, reference, type, name or expression contract is violated.

    Notes:
        Validation is incremental: only usable raw fields and earlier derived fields may
        be referenced. No expressions or generated code are executed.
    """
    if len(features) > max_features:
        raise ValueError(f"Feature budget exceeded: {max_features}")
    types = {name: field.dtype for name, field in schema.usable.items()}
    reserved = {f.name for f in schema.fields}
    signatures = set()
    for feature in features:
        FeatureSpec.model_validate(feature)
        if feature.name in reserved:
            raise ValueError(f"Feature name already exists or is excluded: {feature.name}")
        for name, expected in zip(feature.inputs, OPS[feature.op]):
            if name not in types:
                raise ValueError(f"Unknown, excluded or forward reference: {name}")
            if expected != "any" and types[name] != expected:
                raise ValueError(f"{feature.op} expects {expected}, got {types[name]} for {name}")
        if feature.signature in signatures:
            raise ValueError("Duplicate feature expression")
        signatures.add(feature.signature)
        reserved.add(feature.name)
        types[feature.name] = "categorical" if feature.op == "cross" else "numeric"
    return types
