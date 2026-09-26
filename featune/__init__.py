# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Featune public feature search API.

Exports the supported feature-search API and pinned TabPFN adapters while
keeping heavyweight model and visualization imports lazy.

Created:
    2026-09-21
"""

import logging

from .budget import SearchBudget
from .compiler import FeatureCompiler
from .context import ContextBudget, ContextBuilder
from .evaluation import CVEvaluator
from .fingerprint import ExperimentFingerprint
from .ir import FeatureIR, FeatureSet
from .llm import LLMClient
from .logging_utils import ColorFormatter
from .memory import SearchMemory, SearchMemoryCompressor
from .retrieval import (
    BaseColumnRetriever,
    ColumnSemanticIndex,
    HybridColumnRetriever,
    RandomColumnRetriever,
    RuleBasedColumnRetriever,
    SemanticColumnRetriever,
)
from .samplers import (
    AutonomousSampler,
    BaseFeatureSampler,
    EvolutionSampler,
    HybridSampler,
    LLMSampler,
    RandomSampler,
)
from .schema import DatasetSchema, FeatureSpec, FieldSchema, Hypothesis, Proposal
from .study import FeatureStudy, Trial, create_study, load_study
from .tabpfn import TabPFNClassifier, TabPFNRegressor

__version__ = "1.1.0"
__all__ = [
    "TabPFNClassifier",
    "TabPFNRegressor",
    "ContextBudget",
    "ContextBuilder",
    "SearchMemory",
    "SearchMemoryCompressor",
    "BaseColumnRetriever",
    "ColumnSemanticIndex",
    "SemanticColumnRetriever",
    "RandomColumnRetriever",
    "RuleBasedColumnRetriever",
    "HybridColumnRetriever",
    "SearchBudget",
    "ExperimentFingerprint",
    "FeatureIR",
    "FeatureSet",
    "BaseFeatureSampler",
    "CVEvaluator",
    "DatasetSchema",
    "EvolutionSampler",
    "FeatureCompiler",
    "FeatureSpec",
    "FeatureStudy",
    "FieldSchema",
    "HybridSampler",
    "AutonomousSampler",
    "Hypothesis",
    "LLMClient",
    "ColorFormatter",
    "LLMSampler",
    "Proposal",
    "RandomSampler",
    "Trial",
    "create_study",
    "load_study",
]
logging.getLogger("featune").addHandler(logging.NullHandler())
logging.getLogger("featune").setLevel(logging.INFO)
