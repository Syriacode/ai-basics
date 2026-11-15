from ai_basics.experiments import ECHO_SYSTEM, count_tokens, echo_tokens, show_tokens
from ai_basics.llm import FakeClient, TokenChoice


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
