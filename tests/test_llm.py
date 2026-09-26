# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Protocol, budget, repair and secret-handling checks.

Exercises the module contracts with deterministic local fixtures. External LLM
requests are mocked; temporary storage keeps acceptance runs isolated.

Created:
    2026-09-21
"""

import json

import httpx
import pytest
from test_core import dataset, feature

from featune import LLMClient, LLMSampler
from featune.llm import BudgetExceeded, LLMError, parse_json
from featune.samplers import SearchContext


@pytest.mark.parametrize("provider", ["anthropic", "openai", "local"])
def test_protocol(provider):
    """Verify provider URL/header conventions and known-token accounting without external HTTP.

    Args:
        provider (str): One of anthropic, openai or local; determines URL and payload conventions.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """

    def handler(request):
        """Check protocol request fields and return deterministic provider-shaped usage.

        Args:
            request (httpx.Request): Mock client request; never forwarded to an external provider.

        Returns:
            httpx.Response: HTTP 200 text response in the selected provider protocol.
        """
        body = json.loads(request.content)
        assert body["model"] == "test"
        assert "temperature" not in body
        if provider == "anthropic":
            assert body["max_tokens"] == 2048
            assert str(request.url).endswith("/v1/messages") and request.headers["x-api-key"] == "secret"
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "{}"}],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            )
        assert str(request.url).endswith("/chat/completions")
        assert body["max_completion_tokens" if provider == "openai" else "max_tokens"] == 2048
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    client = LLMClient(
        model="test",
        provider=provider,
        api_key="secret",
        base_url="https://example.com/api",
        transport=httpx.MockTransport(handler),
    )
    assert client.complete("hello") == "{}" and client.usage.total_tokens == 15


def test_custom_openai_compatible_request_parameters():
    """Allow legacy gateways to select their token field and optional temperature."""

    def handler(request):
        body = json.loads(request.content)
        assert body["max_tokens"] == 32 and body["temperature"] == 0.2
        assert "max_completion_tokens" not in body
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client = LLMClient(
        provider="openai",
        model="test",
        api_key="secret",
        base_url="https://example.com/v1",
        max_tokens=32,
        output_token_parameter="max_tokens",
        temperature=0.2,
        transport=httpx.MockTransport(handler),
    )
    assert client.complete("hello") == "{}"


def test_validation_repair_and_no_raw_rows():
    """Verify invalid JSON is repaired within budget without echoing secrets or invalid content.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    _, _, schema = dataset()
    calls = []

    def handler(request):
        """Record prompts and return invalid JSON once, then a valid controlled feature.

        Args:
            request (httpx.Request): Mock client request; never forwarded to an external provider.

        Returns:
            httpx.Response: Deterministic repair fixture with accounted input/output tokens.
        """
        calls.append(json.loads(request.content))
        content = "bad json" if len(calls) == 1 else json.dumps({"features": [feature().model_dump()]})
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": content}],
                "usage": {"input_tokens": 20, "output_tokens": 10},
            },
        )

    sampler = LLMSampler(client=LLMClient(api_key="secret", transport=httpx.MockTransport(handler)))
    context = SearchContext(schema, 1, [], [], "auc", "maximize")
    result = sampler.sample(context)
    assert result.features[0].name == "ratio" and sampler.client.usage.total_tokens == 60
    assert "secret" not in json.dumps(calls) and "bad json" not in json.dumps(calls)
    requests = context.context_metadata["requests"]
    assert len(requests) == 2 and requests[0]["prompt_hash"] != requests[1]["prompt_hash"]
    assert sum(request["usage"]["input_tokens"] for request in requests) == 40
    assert all(
        request["estimated_prompt_tokens"] <= sampler.context_builder.budget.max_tokens
        for request in requests
    )


def test_budget_and_secret_redaction(caplog):
    """Verify preflight budget rejection and sanitized HTTP failure messages/logs.

    Args:
        caplog (pytest.LogCaptureFixture): Fixture capturing logs to verify secret redaction.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    client = LLMClient(api_key="VERY_SECRET", token_budget=1)
    with pytest.raises(BudgetExceeded):
        client.complete("hello")
    client = LLMClient(
        api_key="VERY_SECRET",
        retries=0,
        transport=httpx.MockTransport(lambda req: httpx.Response(401, text="VERY_SECRET")),
    )
    with pytest.raises(LLMError) as error:
        client.complete("hello")
    assert "VERY_SECRET" not in str(error.value) + caplog.text


def test_transport_failure_not_retried():
    """Verify a timed-out request is not resent when billing may already have occurred.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    calls = []

    def handler(request):
        """Count the outgoing attempt and simulate a read timeout.

        Args:
            request (httpx.Request): Mock client request; never forwarded to an external provider.

        Returns:
            None: Always raises httpx.ReadTimeout after recording the request.
        """
        calls.append(1)
        raise httpx.ReadTimeout("private")

    client = LLMClient(api_key="secret", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError, match="not retried"):
        client.complete("hello")
    assert len(calls) == 1 and client.usage.unknown_requests == 1


@pytest.mark.parametrize("content", ['{"a":1,"a":2}', '{"x":NaN}', 'text {"features":[]}'])
def test_ambiguous_json_rejected(content):
    """Verify duplicate keys, non-finite constants and prose-wrapped JSON are rejected.

    Args:
        content (str): Provider response text expected to contain one JSON value.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    with pytest.raises(ValueError):
        parse_json(content)


def test_hallucination_rejected_with_tokens_saved(tmp_path):
    """Verify unknown-field proposals fail while billed repair tokens remain durable.

    Args:
        tmp_path (pathlib.Path): Pytest-managed temporary directory isolating persisted artifacts.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    from sklearn.linear_model import LogisticRegression

    from featune import CVEvaluator, create_study

    X, y, schema = dataset()
    bad = feature(inputs=["hallucination", "income"]).model_dump()

    def handler(request):
        """Return an unknown-field proposal with billable mock usage.

        Args:
            request (httpx.Request): Mock client request; never forwarded to an external provider.

        Returns:
            httpx.Response: HTTP 200 containing a proposal expected to fail DSL validation.
        """
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": json.dumps({"features": [bad]})}],
                "usage": {"input_tokens": 8, "output_tokens": 4},
            },
        )

    sampler = LLMSampler(client=LLMClient(api_key="secret", transport=httpx.MockTransport(handler)))
    study = create_study(sampler=sampler, storage=tmp_path).optimize(
        X, y, schema, evaluator=CVEvaluator(LogisticRegression(), cv=2, importance_repeats=0), n_trials=1
    )
    assert study.trials[1].state == "FAIL" and study.trials[1].tokens["input_tokens"] == 16
    assert "secret" not in (tmp_path / "default" / "search.log").read_text()


def test_unknown_usage_and_cached_tokens():
    """Verify cached input accounting and unknown-usage tracking on malformed responses.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """

    def handler(request):
        """Return a protocol response reporting normal and cached input token buckets.

        Args:
            request (httpx.Request): Mock client request; never forwarded to an external provider.

        Returns:
            httpx.Response: Deterministic usage fixture for aggregate billing assertions.
        """
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "{}"}],
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 10,
                    "cache_creation_input_tokens": 3,
                },
            },
        )

    client = LLMClient(api_key="secret", transport=httpx.MockTransport(handler))
    client.complete("hello")
    assert client.usage.total_tokens == 20 and client.usage.unknown_requests == 0
    broken = LLMClient(
        api_key="secret", transport=httpx.MockTransport(lambda req: httpx.Response(200, text="invalid"))
    )
    with pytest.raises(LLMError, match="Malformed"):
        broken.complete("hello")
    assert broken.usage.unknown_requests == 1


def test_empty_openai_choices_become_a_failed_trial():
    """Verify empty choices become recorded FAIL trials rather than aborting the study.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    from sklearn.linear_model import LogisticRegression

    from featune import CVEvaluator, create_study

    client = LLMClient(
        provider="local",
        model="test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 1}}
            )
        ),
    )
    X, y, schema = dataset()
    study = create_study(sampler=LLMSampler(client=client)).optimize(
        X,
        y,
        schema,
        evaluator=CVEvaluator(LogisticRegression(), cv=2, importance_repeats=0),
        n_trials=2,
        refit=False,
    )
    assert [trial.state for trial in study.trials] == ["COMPLETE", "FAIL", "FAIL"]
    assert all(trial.error == "Malformed LLM protocol response" for trial in study.trials[1:])
    assert client.usage.total_tokens == 6


def test_llm_per_trial_limit_is_enforced():
    """Verify a valid-looking proposal cannot exceed its requested feature-count ceiling.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    _, _, schema = dataset()
    proposal = {"features": [feature().model_dump(), feature("square", "square", ["loan"]).model_dump()]}

    def handler(request):
        """Return more valid expressions than the configured per-trial feature cap.

        Args:
            request (httpx.Request): Mock client request; never forwarded to an external provider.

        Returns:
            httpx.Response: HTTP 200 response expected to be rejected by proposal-size validation.
        """
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": json.dumps(proposal)}],
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        )

    sampler = LLMSampler(
        features_per_trial=1,
        repair_attempts=0,
        client=LLMClient(api_key="secret", transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(LLMError, match="Invalid feature proposal"):
        sampler.sample(SearchContext(schema, 1, [], [], "auc", "maximize"))


def test_context_is_bounded_and_statistics_contain_no_rows():
    """Verify history trimming respects context caps without leaking client credentials.

    Returns:
        None: Assertion success is the test result; failures raise AssertionError.
    """
    _, _, schema = dataset()
    prompts = []

    def handler(request):
        """Capture the bounded prompt and return a deterministic valid ratio proposal.

        Args:
            request (httpx.Request): Mock client request; never forwarded to an external provider.

        Returns:
            httpx.Response: Mock response with known token usage.
        """
        prompts.append(json.loads(request.content)["messages"][0]["content"])
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": json.dumps({"features": [feature().model_dump()]})}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    sampler = LLMSampler(
        max_context_chars=8000, client=LLMClient(api_key="secret", transport=httpx.MockTransport(handler))
    )
    context = SearchContext(
        schema,
        1,
        [],
        [{"number": i, "state": "FAIL", "error": "x" * 3000} for i in range(10)],
        "auc",
        "maximize",
        column_statistics={"loan": {"missing_rate": 0.0}},
    )
    sampler.sample(context)
    assert len(prompts[0]) <= 8000 and '"column_statistics"' in prompts[0]
    assert "secret" not in prompts[0]
    sampler.max_context_chars = 1000
    with pytest.raises(LLMError, match="max_context_chars"):
        sampler.sample(context)
