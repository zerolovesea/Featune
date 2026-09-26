# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Persistable search budgets with explicit consumption units.

Separates cumulative search resource ceilings from per-request context limits.

Created:
    2026-09-21
"""

from pydantic import Field

from .schema import Contract


class SearchBudget(Contract):
    """Define cumulative resource ceilings, independent of prompt context limits.

    Attributes:
        max_trials (int or None): Candidate attempts, including failed and duplicate proposals.
        max_model_fits (int or None): Model-fit attempts, including baseline and final refit.
        max_wall_time (float or None): Active optimize wall-clock seconds across calls.
        max_llm_tokens (int or None): Known cumulative input plus output tokens.
        max_llm_cost (float or None): Estimated API currency units; token prices are required.
        max_valid_proposals (int or None): Proposals that pass schema and parameter validation.

    Notes:
        None disables a ceiling. Time is checked between work units, not by interrupting
        an in-progress third-party fit or HTTP call.
    """

    max_trials: int | None = Field(default=None, ge=0)
    max_model_fits: int | None = Field(default=None, ge=0)
    max_wall_time: float | None = Field(default=None, gt=0)
    max_llm_tokens: int | None = Field(default=None, ge=0)
    max_llm_cost: float | None = Field(default=None, ge=0)
    max_valid_proposals: int | None = Field(default=None, ge=0)

    def exhausted(self, usage: dict) -> str | None:
        """Find the first configured ceiling reached by current consumption.

        Args:
            usage (dict): Consumed resources keyed without the max_ prefix; absent entries count as zero.

        Returns:
            str or None: Exhausted max_ field name, or None when all limits permit work.
        """
        for name, limit in self.model_dump().items():
            if limit is not None and usage.get(name.removeprefix("max_"), 0) >= limit:
                return name
        return None

    def remaining(self, usage: dict) -> dict:
        """Calculate nonnegative remaining capacity for configured ceilings.

        Args:
            usage (dict): Consumed resources keyed without the max_ prefix; absent entries count as zero.

        Returns:
            dict[str, int or float]: Resource names without max_; unlimited resources are omitted.
        """
        return {
            name.removeprefix("max_"): max(0, limit - usage.get(name.removeprefix("max_"), 0))
            for name, limit in self.model_dump().items()
            if limit is not None
        }
