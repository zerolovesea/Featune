# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Bounded OpenAI/Anthropic-compatible JSON transport.

Handles protocol transport and billing uncertainty only. Proposal semantics are
validated separately; response text is never executed or logged wholesale.

Created:
    2026-09-21
"""

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger("featune")


class LLMError(RuntimeError):
    """Signal a sanitized remote/protocol/proposal failure safe for trial persistence.

    Notes:
        Response bodies and credentials must not be included in exception messages.
    """


class BudgetExceeded(LLMError):
    """Signal that another request or model fit cannot safely fit its resource budget.

    Notes:
        FeatureStudy records a normal budget stop rather than continuing to spend.
    """

    pass


@dataclass
class TokenUsage:
    """Track known billing totals separately from requests with uncertain usage.

    Attributes:
        input_tokens (int): Cumulative known prompt tokens, including reported cached input.
        output_tokens (int): Cumulative known completion tokens.
        requests (int): HTTP request attempts, including retries and transport failures.
        unknown_requests (int): Requests whose billing could not be established from a response.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    unknown_requests: int = 0

    @property
    def total_tokens(self):
        """Sum known prompt and completion usage.

        Returns:
            int: Known token count; excludes unmeasurable usage from unknown requests.
        """
        return self.input_tokens + self.output_tokens

    def to_dict(self):
        """Serialize billing counters for study recovery.

        Returns:
            dict[str, int]: Detached usage fields.
        """
        return asdict(self)


class LLMClient:
    """Send bounded JSON-generation requests over supported chat protocols.

    Attributes:
        usage (TokenUsage): Mutable cumulative accounting restored when a study resumes.
        token_budget (int or None): Optional cumulative known-token ceiling.
        cost_budget (float or None): Optional cumulative estimated-cost ceiling.

    Notes:
        Credentials remain in memory. Nonlocal endpoints require HTTPS; redirects are
        disabled. Budgeted requests stop after unknown billing instead of guessing usage.
        See __init__ for endpoint, pricing, retry and generation settings.
    """

    def __init__(
        self,
        model=None,
        provider="anthropic",
        base_url=None,
        api_key=None,
        temperature=None,
        max_tokens=2048,
        output_token_parameter=None,
        timeout=60.0,
        retries=2,
        token_budget=None,
        input_cost_per_million=None,
        output_cost_per_million=None,
        cost_budget=None,
        transport=None,
    ):
        """Resolve endpoint credentials and validate request/billing limits.

        Args:
            model (str or None): Provider model identifier; None uses FEATUNE_MODEL or the client default.
            provider (str): One of anthropic, openai or local; determines URL and payload conventions.
            base_url (str or None): Provider endpoint; defaults to FEATUNE_BASE_URL or provider default.
            api_key (str or None): Credential read from FEATUNE_API_KEY when omitted; never persisted.
            temperature (float or None): Sampling temperature; None uses the provider default.
            max_tokens (int): Maximum completion tokens requested from the provider; must be positive.
            output_token_parameter (str or None): OpenAI-compatible output limit field; defaults to
                max_completion_tokens for OpenAI and max_tokens for local servers.
            timeout (float): Positive HTTP timeout in seconds.
            retries (int): Maximum retry count for eligible transient HTTP failures.
            token_budget (int or None): Cumulative known input/output token ceiling; None is unlimited.
            input_cost_per_million (float or None): Currency units per million input tokens, including
                cached input.
            output_cost_per_million (float or None): Currency units per million completion tokens.
            cost_budget (float or None): Cumulative cost ceiling; requires both token prices.
            transport (httpx.BaseTransport or None): Optional injectable HTTP transport, primarily for
                tests.

        Raises:
            ValueError: Provider, endpoint, credential, request limit or price configuration is invalid.

        Notes:
            The local provider bypasses mandatory API-key validation but still requires an
            explicit appropriate model and usually a local base_url.
        """
        if provider not in {"anthropic", "openai", "local"}:
            raise ValueError("provider must be anthropic, openai or local")
        default_url = "https://api.anthropic.com" if provider == "anthropic" else "https://api.openai.com/v1"
        self.base_url = (base_url or os.getenv("FEATUNE_BASE_URL") or default_url).rstrip("/")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or parsed.query:
            raise ValueError("Invalid base_url; credentials and query strings are not allowed")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Non-local endpoints require HTTPS")
        self.api_key = api_key or os.getenv("FEATUNE_API_KEY")
        if provider != "local" and not self.api_key:
            raise ValueError("Set FEATUNE_API_KEY or pass api_key")
        if (
            max_tokens < 1
            or retries < 0
            or timeout <= 0
            or (temperature is not None and not 0 <= temperature <= 2)
        ):
            raise ValueError("Invalid request limits")
        if output_token_parameter is None:
            output_token_parameter = "max_completion_tokens" if provider == "openai" else "max_tokens"
        if output_token_parameter not in {"max_tokens", "max_completion_tokens"} or (
            provider == "anthropic" and output_token_parameter != "max_tokens"
        ):
            raise ValueError("Invalid output token parameter for provider")
        if token_budget is not None and token_budget < 1:
            raise ValueError("token_budget must be positive")
        self.model = model or os.getenv("FEATUNE_MODEL") or "claude-opus-4-8"
        self.provider, self.temperature = provider, temperature
        self.max_tokens, self.timeout, self.retries = max_tokens, timeout, retries
        self.output_token_parameter = output_token_parameter
        self.token_budget = token_budget
        self.cost_budget = cost_budget
        if any(
            price is not None and (not math.isfinite(price) or price < 0)
            for price in (input_cost_per_million, output_cost_per_million)
        ):
            raise ValueError("Token prices must be nonnegative")
        if cost_budget is not None and (
            not math.isfinite(cost_budget)
            or cost_budget < 0
            or input_cost_per_million is None
            or output_cost_per_million is None
        ):
            raise ValueError("A cost budget requires explicit input and output token prices")
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.transport = transport
        self.usage = TokenUsage()

    @property
    def cost(self):
        """Estimate known request costs using the configured flat token prices.

        Returns:
            float or None: Currency units for known usage; None when either price is absent.

        Notes:
            Unknown requests are not estimated. Provider-specific cache discounts are not
            modeled; inspect usage.unknown_requests and reconcile with the provider bill.
        """
        if self.input_cost_per_million is None or self.output_cost_per_million is None:
            return None
        return (
            self.usage.input_tokens * self.input_cost_per_million
            + self.usage.output_tokens * self.output_cost_per_million
        ) / 1_000_000

    def complete(self, prompt):
        # UTF-8 byte length is a conservative preflight reserve, not billed usage.
        """Send one bounded completion operation and extract response text.

        Args:
            prompt (str): Fully constructed user prompt; must contain no raw rows or credentials.

        Returns:
            str: Provider text to be separately parsed and validated as a Proposal.

        Raises:
            BudgetExceeded: The next request cannot be reserved, or budgeted retry billing is unknown.
            LLMError: Transport, HTTP, response shape or content-size validation failed.

        Notes:
            Mutates usage for every HTTP attempt, including failures. A transport failure is
            never blindly retried because the server may already have billed it. Selected
            HTTP failures have bounded retries. A successful text response is not yet a
            validated feature proposal; LLMSampler owns JSON/DSL validation.
        """
        reserve = len(prompt.encode("utf-8")) + self.max_tokens
        if self.token_budget is not None and (
            self.usage.unknown_requests or self.usage.total_tokens + reserve > self.token_budget
        ):
            raise BudgetExceeded("Token budget cannot safely reserve the next request")
        if self.cost_budget is not None:
            if self.cost is None:
                raise ValueError("A cost budget requires explicit token prices")
            reserve_cost = (
                len(prompt.encode("utf-8")) * self.input_cost_per_million
                + self.max_tokens * self.output_cost_per_million
            ) / 1_000_000
            if self.usage.unknown_requests or self.cost + reserve_cost > self.cost_budget:
                raise BudgetExceeded("API cost budget cannot safely reserve the next request")
        anthropic = self.provider == "anthropic"
        suffix = "/v1/messages" if anthropic else "/chat/completions"
        url = self.base_url if self.base_url.endswith(suffix) else self.base_url + suffix
        headers = {"Content-Type": "application/json", "User-Agent": "Featune/1.0"}
        if anthropic:
            headers.update({"anthropic-version": "2023-06-01", "x-api-key": self.api_key})
        elif self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        payload = {
            "model": self.model,
            self.output_token_parameter: self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        for attempt in range(self.retries + 1):
            logger.info("LLM request %s | model=%s | attempt=%d", self.provider, self.model, attempt + 1)
            try:
                with httpx.Client(
                    timeout=self.timeout, transport=self.transport, follow_redirects=False
                ) as client:
                    response = client.post(url, headers=headers, json=payload)
            except httpx.TransportError:
                # A disconnected response may already have been billed. Never silently resend it.
                self.usage.requests += 1
                self.usage.unknown_requests += 1
                raise LLMError("LLM transport failed; billing unknown, request not retried") from None
            self.usage.requests += 1
            if response.status_code in {429, 502, 503, 504} and attempt < self.retries:
                if response.status_code != 429:
                    self.usage.unknown_requests += 1
                    if self.token_budget is not None or self.cost_budget is not None:
                        raise BudgetExceeded("Remote failure with unknown billing; budgeted run stopped")
                time.sleep(min(2**attempt, 8))
                continue
            if response.is_error:
                if response.status_code >= 500:
                    self.usage.unknown_requests += 1
                raise LLMError(f"LLM HTTP {response.status_code}; response body omitted")
            # Clear this flag only after complete usage arrives; valid billing survives bad content.
            self.usage.unknown_requests += 1
            try:
                data = response.json()
                usage = data.get("usage", {})
                input_key, output_key = (
                    ("input_tokens", "output_tokens") if anthropic else ("prompt_tokens", "completion_tokens")
                )
                input_tokens = int(usage.get(input_key, 0))
                output_tokens = int(usage.get(output_key, 0))
                if anthropic:
                    input_tokens += int(usage.get("cache_creation_input_tokens", 0)) + int(
                        usage.get("cache_read_input_tokens", 0)
                    )
                if input_tokens < 0 or output_tokens < 0:
                    raise ValueError("Negative token usage")
                self.usage.input_tokens += input_tokens
                self.usage.output_tokens += output_tokens
                if input_key in usage and output_key in usage:
                    self.usage.unknown_requests -= 1
                if anthropic:
                    content = "".join(block["text"] for block in data["content"] if block["type"] == "text")
                else:
                    content = data["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError, AttributeError, OverflowError):
                raise LLMError("Malformed LLM protocol response") from None
            logger.info(
                "LLM usage | input=%d output=%d cumulative=%d unknown_requests=%d",
                input_tokens,
                output_tokens,
                self.usage.total_tokens,
                self.usage.unknown_requests,
            )
            if not isinstance(content, str) or len(content) > 100_000:
                raise LLMError("LLM content is absent or exceeds the response limit")
            return content
        raise LLMError("LLM request retries exhausted")


def parse_json(content):
    """Parse one strict JSON value, optionally enclosed in a json code fence.

    Args:
        content (str): Provider response text expected to contain one JSON value.

    Returns:
        Any: Decoded JSON value; the caller validates its structural schema.

    Raises:
        ValueError: JSON is malformed, keys repeat, or NaN/Infinity appears.

    Notes:
        Does not extract JSON fragments from surrounding prose and never executes text.
    """
    text = content.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()

    def unique_pairs(pairs):
        """Reject duplicate object keys while reconstructing a decoded JSON object.

        Args:
            pairs (list[tuple[str, Any]]): Ordered object members supplied by json.loads.

        Returns:
            dict: Object with each key appearing once.

        Raises:
            ValueError: An object repeats a key.
        """
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        text,
        object_pairs_hook=unique_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")),
    )
