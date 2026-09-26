# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Budget-efficiency curves, target attainment and Pareto summaries.

Computes resource curves and target attainment from inner-search history.
Unknown costs remain unknown rather than being reported as free work.

Created:
    2026-09-22
"""

import pandas as pd


def search_diagnostics(trials, direction="maximize", target_gain=0.005):
    """Build cumulative efficiency curves and inner-CV target-attainment summaries.

    Args:
        trials (sequence[Trial or dict]): Chronological baseline and candidate records.
        direction (str): "maximize" for rewards or "minimize" for losses.
        target_gain (float): Nonnegative improvement over baseline, expressed in the metric's natural
            units.

    Returns:
        tuple[dict, pandas.DataFrame]: Summary metrics and completed-trial trajectory; empty if none
        complete.

    Raises:
        ValueError: target_gain is negative.

    Notes:
        Records must be chronological and include a completed baseline at number zero
        when any trials complete. Candidate evaluations exclude baseline and cache hits;
        fit totals include baseline fits. Unknown billing makes subsequent cost unknown.
        Targets use inner-search gains, never outer-test performance.
    """
    if target_gain < 0:
        raise ValueError("target_gain must be nonnegative")
    records = [trial.to_dict() if hasattr(trial, "to_dict") else trial for trial in trials]
    completed = [trial for trial in records if trial["state"] == "COMPLETE"]
    if not completed:
        return {}, pd.DataFrame()
    baseline = next(trial for trial in completed if trial["number"] == 0)
    sign = 1 if direction == "maximize" else -1
    best = max(completed, key=lambda trial: sign * trial["value"])
    evaluations = fits = tokens = valid = invalid = hits = 0
    elapsed, cost, cost_known = 0.0, 0.0, True
    curve = []
    for trial in records:
        if trial["number"]:
            valid += int(trial.get("valid_proposal", False))
            invalid += int(not trial.get("valid_proposal", False))
            hits += int(trial.get("cache_hit", False))
            evaluations += int(
                not trial.get("cache_hit", False)
                and (trial.get("model_fits", 0) > 0 or trial["state"] == "COMPLETE")
            )
        fits += trial.get("model_fits", 0)
        usage = trial.get("tokens", {})
        tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        elapsed = trial.get("elapsed") or elapsed + trial.get("duration", 0.0)
        trial_cost = trial.get("api_cost")
        if usage.get("unknown_requests", 0):
            cost_known = False
        if trial_cost is not None:
            cost += trial_cost
        elif usage.get("requests", 0) or usage.get("unknown_requests", 0):
            cost_known = False
        if trial["state"] == "COMPLETE":
            curve.append(
                {
                    "trial": trial["number"],
                    "value": trial["value"],
                    "evaluations": evaluations,
                    "model_fits": fits,
                    "wall_time": elapsed,
                    "tokens": tokens,
                    "api_cost": cost if cost_known else None,
                    "gain": sign * (trial["value"] - baseline["value"]),
                }
            )
    frame = pd.DataFrame(curve)
    best_row = next(row for row in curve if row["trial"] == best["number"])
    attempts = valid + invalid
    summary = {
        "candidate_evaluations": evaluations,
        "model_fits": fits,
        "evaluations_to_best": best_row["evaluations"],
        "time_to_best": best_row["wall_time"],
        "cost_to_best": best_row["api_cost"],
        "tokens": tokens,
        "api_cost": cost if cost_known else None,
        "valid_proposal_rate": valid / attempts if attempts else 0.0,
        "invalid_proposal_rate": invalid / attempts if attempts else 0.0,
        "cache_hit_rate": hits / valid if valid else 0.0,
    }
    best_gain = sign * (best["value"] - baseline["value"])
    for name, gain in [("fixed_gain", target_gain), ("95pct_final_gain", 0.95 * best_gain)]:
        reached = next((row for row in curve if row["gain"] >= gain), None) if gain > 0 else None
        summary[name] = {
            "target_gain": gain,
            "reached": reached is not None,
            "evaluations_to_X": reached["evaluations"] if reached else None,
            "time_to_X": reached["wall_time"] if reached else None,
            "cost_to_X": reached["api_cost"] if reached else None,
        }
    return summary, frame


def pareto_frontier(curve, direction="maximize", resource="evaluations"):
    """Keep score improvements along an ascending resource-cost axis.

    Args:
        curve (pandas.DataFrame): search_diagnostics trajectory with value and the chosen resource.
        direction (str): "maximize" for rewards or "minimize" for losses.
        resource (str): evaluations, model_fits, wall_time, tokens or api_cost for the Pareto x-axis.

    Returns:
        pandas.DataFrame: Undominated rows without the temporary utility column.

    Raises:
        ValueError: resource is not supported.

    Notes:
        Rows with unknown resource values are excluded. Equal-cost ties keep the best
        utility before checking dominance; the input frame is not modified.
    """
    if resource not in {"evaluations", "model_fits", "wall_time", "api_cost", "tokens"}:
        raise ValueError("Unsupported Pareto resource")
    if curve.empty:
        return curve
    sign = 1 if direction == "maximize" else -1
    ordered = curve.dropna(subset=[resource]).copy()
    ordered["utility"] = sign * ordered["value"]
    ordered = ordered.sort_values([resource, "utility"], ascending=[True, False])
    keep, best = [], float("-inf")
    for index, row in ordered.iterrows():
        if row.utility > best:
            keep.append(index)
            best = row.utility
    return ordered.loc[keep].drop(columns="utility")
