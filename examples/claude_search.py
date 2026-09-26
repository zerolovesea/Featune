# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Anthropic-compatible semantic search with bounded context.

Demonstrates explicitly invoked, potentially billable semantic search.
Credentials are read from environment configuration and never written to artifacts.

Created:
    2026-09-21
"""

import argparse
import json
import logging
from pathlib import Path

from quickstart import make_dataset
from sklearn.model_selection import train_test_split

import featune


def main():
    """Run an explicitly invoked, credentialed Anthropic-compatible search example.

    Returns:
        None: Prints and saves metrics, pipeline, summary and HTML report.

    Notes:
        Reads command-line options and FEATUNE_* client configuration. This function
        may incur API charges. Outer-test AUC is calculated only after optimization,
        and raw credentials/provider responses are not written to artifacts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", default="runs/claude")
    parser.add_argument("--guidance-file", type=Path, help="UTF-8 hypotheses to explore, sent to the LLM")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    X, y, schema = make_dataset(n=300)
    if args.guidance_file:
        schema = featune.DatasetSchema.model_validate(
            {**schema.model_dump(), "search_guidance": args.guidance_file.read_text(encoding="utf-8")}
        )
    train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y, random_state=42)
    sampler = featune.LLMSampler(provider="anthropic", max_tokens=2048, token_budget=100_000)
    study = featune.create_study(sampler=sampler, storage=args.output, study_name="semantic")
    study.optimize(
        train_x,
        train_y,
        schema,
        evaluator=featune.CVEvaluator(cv=3),
        n_trials=args.trials,
    )
    from sklearn.metrics import roc_auc_score

    summary = {
        "baseline_cv_auc": study.baseline_trial.value,
        "best_cv_auc": study.best_value,
        "test_auc": roc_auc_score(test_y, study.pipeline_.predict_proba(test_x)[:, 1]),
        "features": [feature.model_dump() for feature in study.best_features],
        "usage": sampler.client.usage.to_dict(),
        "states": [t.state for t in study.trials],
    }
    Path(args.output, "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    study.export_pipeline(Path(args.output, "pipeline.joblib"))
    study.report(Path(args.output, "report.html"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
