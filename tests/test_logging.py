# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Live log color selection checks."""

import logging

from featune.logging_utils import ColorFormatter


def test_color_formatter_separates_live_sections_and_preserves_plain_text():
    """Color proposals, judgments and metrics without changing stored messages."""
    colored = ColorFormatter("%(message)s")
    plain = ColorFormatter("%(message)s", color=False)
    cases = {
        "Trial 1 | hypothesis_proposed | {}": "36",
        'Trial 1 | hypothesis_assessed | {"status": "rejected"}': "31",
        'Trial 1 | hypothesis_assessed | {"status": "supported"}': "32",
        'Trial 1 | hypothesis_assessed | {"status": "inconclusive"}': "33",
        "LLM usage | input=100 output=10": "35",
        "CV fold 1/3 | auc=0.8": "34",
        "Trial 1 | COMPLETE | decision=selected_best": "32",
        "Trial 2 | COMPLETE | decision=kept_not_best": "34",
    }
    for message, code in cases.items():
        record = logging.LogRecord("featune", logging.INFO, __file__, 1, message, (), None)
        assert colored.format(record) == f"\x1b[{code}m{message}\x1b[0m"
        assert plain.format(record) == message
        assert record.getMessage() == message
    other = logging.LogRecord("httpx", logging.INFO, __file__, 1, "HTTP Request", (), None)
    assert colored.format(other) == "HTTP Request"
