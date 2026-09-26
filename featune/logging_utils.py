# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Optional colors for live Featune logs; stored logs remain plain text."""

import logging


class ColorFormatter(logging.Formatter):
    """Color live log sections without changing their stored messages."""

    def __init__(self, fmt, *, color=True):
        """Set the output format and whether ANSI colors are enabled."""
        super().__init__(fmt)
        self.color = color

    def format(self, record):
        """Format a record and color Featune sections when requested."""
        line = super().format(record)
        if not self.color or record.name != "featune":
            return line
        message = record.getMessage()
        if "hypothesis_assessed" in message:
            code = (
                "31"
                if '"status": "rejected"' in message
                else "32"
                if '"status": "supported"' in message or '"status": "strongly_supported"' in message
                else "33"
            )
        elif "hypothesis_proposed" in message:
            code = "36"
        elif message.startswith("LLM "):
            code = "35"
        elif message.startswith("CV fold") or message.startswith("Baseline | COMPLETE"):
            code = "34"
        elif "decision=selected_best" in message:
            code = "32"
        elif " | COMPLETE | decision=" in message:
            code = "34"
        elif record.levelno >= logging.WARNING or "BUDGET_EXCEEDED" in message:
            code = "33"
        else:
            return line
        return f"\x1b[{code}m{line}\x1b[0m"
