# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Configuration-driven search and artifact commands.

Exposes a closed configuration-driven command surface. Credentialed LLM work is
performed only by an explicitly configured optimize command.

Created:
    2026-09-21
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge

from .evaluation import CVEvaluator
from .logging_utils import ColorFormatter
from .samplers import AutonomousSampler, EvolutionSampler, HybridSampler, LLMSampler, RandomSampler
from .schema import DatasetSchema
from .study import create_study, load_study
from .tabpfn import TabPFNClassifier, TabPFNRegressor, default_estimator


def build_estimator(config, metric="auc", random_state=42):
    """Instantiate an estimator from the CLI's closed model registry.

    Args:
        config (dict): type (default tabpfn) and optional params mapping.
        metric (str): Metric identifying the task for the automatic TabPFN preset.
        random_state (int): Seed for the default TabPFN preset.

    Returns:
        sklearn estimator: Unfitted model with the requested constructor parameters.

    Raises:
        ValueError: The estimator type is not registered.
        ImportError: A required model dependency is missing from the installation.
    """
    classes = {
        "logistic": LogisticRegression,
        "ridge": Ridge,
        "hist_classifier": HistGradientBoostingClassifier,
        "hist_regressor": HistGradientBoostingRegressor,
        "random_forest": RandomForestClassifier,
    }
    classes.update(tabpfn_classifier=TabPFNClassifier, tabpfn_regressor=TabPFNRegressor)
    kind = config.get("type", "tabpfn")
    if kind == "tabpfn":
        return default_estimator(metric, random_state).set_params(**config.get("params", {}))
    if kind.startswith("torch_"):
        from .torch import TorchClassifier, TorchRegressor

        classes.update(torch_classifier=TorchClassifier, torch_regressor=TorchRegressor)
    if kind not in classes:
        raise ValueError(f"Unknown estimator: {kind}")
    return classes[kind](**config.get("params", {}))


def build_sampler(config):
    """Instantiate a configured controlled sampler while rejecting inline credentials.

    Args:
        config (dict): type plus sampler constructor settings; hybrid has llm/random sub-configs.

    Returns:
        BaseFeatureSampler: Random, evolution, LLM or hybrid policy.

    Raises:
        ValueError: The type is unknown, an API key is embedded, or sampler settings are invalid.

    Notes:
        LLM credentials must come from the environment; constructing a client does not
        issue an HTTP request. Model generation starts only during optimize.
    """
    options = dict(config)
    kind = options.pop("type", "random")
    if "api_key" in options or "api_key" in options.get("llm", {}):
        raise ValueError("Use FEATUNE_API_KEY instead of saving api_key in CLI configuration")
    classes = {"random": RandomSampler, "evolution": EvolutionSampler, "llm": LLMSampler}
    if kind == "hybrid":
        return HybridSampler(
            llm_sampler=LLMSampler(**options.pop("llm", {})),
            random_sampler=EvolutionSampler(**options.pop("random", {})),
            **options,
        )
    if kind == "autonomous":
        llm = options.pop("llm", None)
        numeric = options.pop("numeric", {})
        if llm is not None:
            llm.setdefault("features_per_trial", 1)
        return AutonomousSampler(
            llm_sampler=LLMSampler(**llm) if llm is not None else None,
            numeric_sampler=RandomSampler(**numeric),
            **options,
        )
    if kind not in classes:
        raise ValueError(f"Unknown sampler: {kind}")
    return classes[kind](**options)


def main(argv=None):
    """Dispatch configuration-driven optimization, inspection, reporting or feature export.

    Args:
        argv (sequence[str] or None): Command-line arguments; None reads sys.argv[1:].

    Returns:
        None: Writes command output and any requested artifacts.

    Raises:
        SystemExit: argparse handles help or invalid command arguments.

    Notes:
        Configuration paths resolve relative to the config file. optimize can perform
        network requests when an LLM sampler is explicitly configured; repeated runs
        add attempts rather than acting as idempotent queries. Errors from data loading,
        validation, fitting and persistence propagate to the command caller.
    """
    parser = argparse.ArgumentParser(prog="featune", description="Semantic feature engineering search")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    optimize = commands.add_parser("optimize", help="Run or resume from a JSON configuration")
    optimize.add_argument("config", type=Path)
    for command in ("report", "export", "inspect"):
        sub = commands.add_parser(command)
        sub.add_argument("--storage", required=True)
        sub.add_argument("--study-name", default="default")
        if command != "inspect":
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
        handler = logging.getLogger().handlers[0]
        handler.setFormatter(
            ColorFormatter("%(asctime)s | %(levelname)s | %(message)s", color=handler.stream.isatty())
        )
    if args.command == "optimize":
        config = json.loads(args.config.read_text(encoding="utf-8"))
        base = args.config.resolve().parent
        dataset = config["data"]
        path = base / dataset["path"]
        data = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        y = data.pop(dataset["target"])
        groups = data.pop(dataset["groups"]) if dataset.get("groups") else None
        schema_config = config["schema"]
        if isinstance(schema_config, str):
            schema_config = json.loads((base / schema_config).read_text(encoding="utf-8"))
        schema = DatasetSchema.model_validate(schema_config)
        if dataset["target"] in schema.usable:
            raise ValueError("Target must not be a usable schema field")
        options = dict(config.get("study", {}))
        sampler = build_sampler(
            config.get(
                "sampler", {"type": "autonomous"} if options.get("search_strategy") == "autonomous" else {}
            )
        )
        options["storage"] = base / options.get("storage", "runs")
        study = create_study(sampler=sampler, **options)
        evaluation = dict(config.get("evaluation", {}))
        evaluation.setdefault("metric", study.metric)
        estimator = build_estimator(config.get("estimator", {}), evaluation["metric"], study.random_state)
        evaluator = CVEvaluator(estimator, **evaluation)
        study.optimize(data, y, schema, evaluator=evaluator, groups=groups, **config.get("optimize", {}))
        destination = study.storage.directory
        completed = any(t.state == "COMPLETE" for t in study.trials)
        if hasattr(study, "pipeline_"):
            study.export_pipeline(destination / "pipeline.joblib")
        if completed:
            study.export_features(destination / "features.json")
        study.trials_dataframe().to_json(destination / "trials.json", orient="records", indent=2)
        if completed and config.get("report", False):
            study.report(destination / "report.html")
        print(
            json.dumps(
                {
                    "best_value": study.best_value if completed else None,
                    "best_trial": study.best_trial.number if completed else None,
                    "stop_reason": study.stop_reason,
                    "directory": str(destination),
                }
            )
        )
    else:
        study = load_study(args.storage, args.study_name)
        if args.command == "report":
            study.report(args.output)
        elif args.command == "export":
            study.export_features(args.output)
        else:
            print(study.trials_dataframe().drop(columns=["features", "hypotheses"]).to_string(index=False))
