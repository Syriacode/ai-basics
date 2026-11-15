import pytest

from ai_basics.experiments import (
    ECHO_SYSTEM,
    count_tokens,
    echo_tokens,
    next_token_table,
    show_tokens,
)
from ai_basics.llm import FakeClient, LLMResponse, TokenChoice


def choices(*tokens):
    return [TokenChoice(token=t, logprob=0.0, top=[]) for t in tokens]


def test_show_tokens_marks_spaces():
    assert show_tokens(choices("The", " cat", " sat", ".")) == "The|·cat|·sat|."


def test_show_tokens_marks_newlines_and_repeated_spaces():
    assert show_tokens(choices("a", "\n", "  x")) == "a|\\n|··x"


def test_show_tokens_empty():
    assert show_tokens([]) == ""


def test_count_tokens_reads_input_tokens_from_a_raw_one_token_completion():
    client = FakeClient()

    assert count_tokens(client, "The cat sat.") == 100  # FakeClient reports 100 input tokens
    (call,) = client.calls
    assert call.method == "complete"
    assert call.prompt == "The cat sat."
    assert call.num_predict == 1


def test_echo_tokens_asks_for_an_exact_repeat_with_logprobs():
    client = FakeClient()

    tokens = echo_tokens(client, "unbelievably")

    (call,) = client.calls
    assert call.method == "chat"
    assert call.messages == [
        {"role": "system", "content": ECHO_SYSTEM},
        {"role": "user", "content": "unbelievably"},
    ]
    assert call.top_logprobs == 1
    assert [t.token for t in tokens] == ["log"]  # FakeClient's one paper-example token


def test_next_token_table_returns_token_probability_pairs():
    client = FakeClient()

    table = next_token_table(client, "Binary search runs in O(")

    assert [token for token, _ in table] == ["log", "n", "1"]
    assert [p for _, p in table] == pytest.approx([0.665, 0.245, 0.090], abs=0.001)
    (call,) = client.calls
    assert call.method == "complete"
    assert call.prompt == "Binary search runs in O("
    assert call.num_predict == 1
    assert call.top_logprobs == 5


def test_next_token_table_passes_k_through():
    client = FakeClient()

    next_token_table(client, "Hello", k=3)

    assert client.calls[0].top_logprobs == 3


class NoTokensClient:
    """A stand-in model that returns no logprobs, like an old Ollama."""

    model = "stub"

    def complete(self, prompt, **kwargs):
        return LLMResponse(text="", model="stub", input_tokens=1, output_tokens=0, latency_ms=0.0)


def test_next_token_table_is_empty_when_there_are_no_tokens():
    assert next_token_table(NoTokensClient(), "Hello") == []
