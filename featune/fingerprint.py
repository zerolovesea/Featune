# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Auditable experiment identity and resume compatibility.

Captures source, dependency, data and configuration identities without persisting
raw observations. Source comments/docstrings currently affect compatibility.

Created:
    2026-09-21
"""

import importlib.metadata
import platform
import subprocess
from pathlib import Path

import joblib

from .context import CONTEXT_VERSION
from .ir import COMPILER_VERSION, DSL_VERSION, PROMPT_VERSION, content_hash
from .memory import MEMORY_VERSION
from .retrieval import RETRIEVAL_VERSION
from .schema import Contract
from .tabpfn import _TabPFNBase


class ExperimentFingerprint(Contract):
    """Capture numerical, code and proposal identity for safe resume decisions.

    Attributes:
        featune_version (str): Installed package version.
        git_commit (str or None): Repository revision when available.
        python_version (str): Python interpreter version.
        dependencies (dict[str, str]): Numerical/validation/client dependency versions.
        source_hash (str): Digest of package Python source text, including comments/docstrings.
        dataset (str): Dataset and group fingerprint.
        schema_fingerprint (str): Full semantic schema hash.
        cv (str): Splits, metrics and explanation-configuration fingerprint.
        estimator (str): Cloneable estimator configuration hash.
        model_provenance (dict): Immutable TabPFN checkpoint identity, empty for other estimators.
        dsl_version (str): Controlled expression format version.
        compiler_version (str): Numerical compiler semantics version.
        prompt_version (str): Prompt template version.
        context_version (str): Context construction version.
        memory_version (str): Search-memory compression version.
        retrieval_version (str): Retrieval algorithm version.
        llm_provider (str or None): Remote/local protocol family.
        llm_model (str or None): Configured provider model identifier.
        seed (int): Study seed.
        sampler (dict): Serializable sampler and context settings.
        configuration (str): Hash of study behavior settings.

    Notes:
        git_commit is informational and excluded from compatibility_hash. source_hash
        currently includes documentation edits, so comment-only source changes also
        invalidate resume compatibility. Dataset bytes and credentials are not stored.
    """

    featune_version: str
    git_commit: str | None
    python_version: str
    dependencies: dict[str, str]
    source_hash: str
    dataset: str
    schema_fingerprint: str
    cv: str
    estimator: str
    model_provenance: dict = {}
    dsl_version: str = DSL_VERSION
    compiler_version: str = COMPILER_VERSION
    prompt_version: str = PROMPT_VERSION
    context_version: str = CONTEXT_VERSION
    memory_version: str = MEMORY_VERSION
    retrieval_version: str = RETRIEVAL_VERSION
    llm_provider: str | None = None
    llm_model: str | None = None
    seed: int
    sampler: dict
    configuration: str

    @property
    def compatibility_hash(self):
        # A documentation-only commit does not invalidate numerical results.
        """Hash all compatibility-relevant fields while excluding the Git revision.

        Returns:
            str: Canonical fingerprint used as the resume/cache namespace.
        """
        return content_hash(self.model_dump(exclude={"git_commit"}))

    @classmethod
    def capture(cls, X, y, groups, schema, evaluator, sampler, seed, configuration):
        """Collect reproducibility metadata without persisting raw observations.

        Args:
            X (pandas.DataFrame): Input feature rows with unique columns and schema-compatible types.
            y (array-like): One-dimensional targets aligned with X; never sent to the LLM.
            groups (array-like or None): Group labels aligned with X; groups must not cross CV folds.
            schema (DatasetSchema): Validated field types, semantics, exclusions and task objective.
            evaluator (CVEvaluator): Prepared evaluator with stable splits and estimator configuration.
            sampler (BaseFeatureSampler or None): Proposal policy; None selects the study default.
            seed (int): Seed for deterministic random choices.
            configuration (dict): Serializable settings that affect numerical or proposal compatibility.

        Returns:
            ExperimentFingerprint: Validated experiment identity.

        Notes:
            Requires evaluator.prepare first. Git lookup failures become git_commit=None;
            missing required package metadata and nonserializable configuration errors
            propagate. Optional estimator-package metadata is included when available.
        """
        source = Path(__file__).parent
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=source,
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            commit = result.stdout.strip() if result.returncode == 0 else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            commit = None
        sampler_config = sampler.configuration()
        llm = sampler_config.get("llm") or sampler_config
        dependencies = {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "pydantic", "joblib", "httpx"]
        }
        module = type(evaluator.estimator).__module__.split(".")[0]
        if isinstance(evaluator.estimator, _TabPFNBase):
            module = "tabpfn"
            dependencies["torch"] = importlib.metadata.version("torch")
        elif module == "featune":
            module = "torch"
        try:
            dependencies[module] = importlib.metadata.version(module)
        except importlib.metadata.PackageNotFoundError:
            pass
        return cls(
            featune_version=importlib.metadata.version("featune"),
            git_commit=commit,
            python_version=platform.python_version(),
            dependencies=dependencies,
            source_hash=content_hash({p.name: p.read_text() for p in sorted(source.glob("*.py"))}),
            dataset=joblib.hash((X, y, groups)),
            schema_fingerprint=content_hash(schema.model_dump()),
            cv=joblib.hash(
                (
                    evaluator.splits_,
                    evaluator.metrics,
                    evaluator.random_state,
                    evaluator.importance_repeats,
                    evaluator.importance_max_samples,
                )
            ),
            estimator=joblib.hash(evaluator.estimator),
            model_provenance=(
                evaluator.estimator.provenance() if isinstance(evaluator.estimator, _TabPFNBase) else {}
            ),
            seed=seed,
            sampler=sampler_config,
            configuration=content_hash(configuration),
            llm_provider=llm.get("provider"),
            llm_model=llm.get("model"),
        )
