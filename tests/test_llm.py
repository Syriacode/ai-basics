import json
import math

import httpx
import pytest

from ai_basics.llm import FakeClient, LLMError, OllamaClient, TokenChoice

MESSAGES = [{"role": "user", "content": "What is a token?"}]


def client_with(handler):
    http = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return OllamaClient(http=http)


def recording_handler(requests, reply):
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=reply)

    return handler


CHAT_REPLY = {
    "model": "qwen2.5:3b",
    "message": {"role": "assistant", "content": "A piece of text."},
    "done": True,
    "prompt_eval_count": 12,
    "eval_count": 5,
}

GENERATE_REPLY = {
    "model": "qwen2.5:3b",
    "response": " Paris",
    "done": True,
    "prompt_eval_count": 6,
    "eval_count": 1,
}

LOGPROBS = [
    {
        "token": " Paris",
        "logprob": -0.05,
        "top_logprobs": [
            {"token": " Paris", "logprob": -0.05},
            {"token": " Lyon", "logprob": -4.0},
            {"token": " the", "logprob": -5.5},
        ],
    }
]


def test_default_model_is_qwen():
    assert OllamaClient().model == "qwen2.5:3b"


def test_chat_request_body():
    requests = []
    client = client_with(recording_handler(requests, CHAT_REPLY))

    response = client.chat(MESSAGES)

    assert requests[0].url.path == "/api/chat"
    body = json.loads(requests[0].content)
    assert body["model"] == "qwen2.5:3b"
    assert body["messages"] == MESSAGES
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "num_ctx": 4096}
    assert response.text == "A piece of text."
    assert (response.input_tokens, response.output_tokens) == (12, 5)
    assert response.tokens == []


def test_complete_request_body_is_raw():
    requests = []
    client = client_with(recording_handler(requests, GENERATE_REPLY))

    response = client.complete("The capital of France is", num_predict=1, seed=42)

    assert requests[0].url.path == "/api/generate"
    body = json.loads(requests[0].content)
    assert body["prompt"] == "The capital of France is"
    assert body["raw"] is True
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "num_ctx": 4096, "num_predict": 1, "seed": 42}
    assert response.text == " Paris"
    assert (response.input_tokens, response.output_tokens) == (6, 1)


def test_options_are_passed_through():
    requests = []
    client = client_with(recording_handler(requests, CHAT_REPLY))

    client.chat(MESSAGES, temperature=1.0, num_ctx=1024)

    body = json.loads(requests[0].content)
    assert body["options"] == {"temperature": 1.0, "num_ctx": 1024}


@pytest.mark.parametrize("method", ["chat", "complete"])
def test_no_logprobs_keys_when_not_asked(method):
    requests = []
    reply = CHAT_REPLY if method == "chat" else GENERATE_REPLY
    client = client_with(recording_handler(requests, reply))
    arg = MESSAGES if method == "chat" else "Hello"

    getattr(client, method)(arg)

    body = json.loads(requests[0].content)
    assert "logprobs" not in body
    assert "top_logprobs" not in body


def test_logprobs_requested_and_parsed():
    requests = []
    client = client_with(recording_handler(requests, {**GENERATE_REPLY, "logprobs": LOGPROBS}))

    response = client.complete("The capital of France is", num_predict=1, top_logprobs=3)

    body = json.loads(requests[0].content)
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 3
    assert response.tokens == [
        TokenChoice(
            token=" Paris",
            logprob=-0.05,
            top=[(" Paris", -0.05), (" Lyon", -4.0), (" the", -5.5)],
        )
    ]


def test_missing_logprobs_in_reply_gives_empty_list():
    # An old Ollama ignores the request and sends no "logprobs" field.
    client = client_with(recording_handler([], GENERATE_REPLY))

    assert client.complete("Hi", top_logprobs=3).tokens == []


def test_server_error_becomes_llm_error():
    client = client_with(lambda request: httpx.Response(500))

    with pytest.raises(LLMError):
        client.chat(MESSAGES)
    with pytest.raises(LLMError):
        client.complete("Hi")


def test_connection_error_becomes_llm_error():
    def handler(request):
        raise httpx.ConnectError("ollama is not running")

    with pytest.raises(LLMError):
        client_with(handler).complete("Hi")


def test_fake_client_default_reply_and_recorded_calls():
    fake = FakeClient()

    chat = fake.chat(MESSAGES, temperature=1.0, num_ctx=1024)
    complete = fake.complete("Hi", num_predict=1, seed=7)

    assert chat.text == complete.text == "fake reply"
    assert [c.method for c in fake.calls] == ["chat", "complete"]
    assert fake.calls[0].messages == MESSAGES
    assert (fake.calls[0].temperature, fake.calls[0].num_ctx) == (1.0, 1024)
    assert fake.calls[1].prompt == "Hi"
    assert (fake.calls[1].num_predict, fake.calls[1].seed) == (1, 7)


def test_fake_client_scripted_replies_and_errors():
    fake = FakeClient(["one", LLMError("boom"), "three"])

    assert fake.complete("a").text == "one"
    with pytest.raises(LLMError):
        fake.complete("b")
    assert fake.chat(MESSAGES).text == "three"
    with pytest.raises(AssertionError):
        fake.complete("d")


def test_fake_client_returns_the_paper_example_when_asked():
    response = FakeClient().complete("Binary search runs in O(", top_logprobs=3)

    (choice,) = response.tokens
    assert [token for token, _ in choice.top] == ["log", "n", "1"]
    probs = [math.exp(logprob) for _, logprob in choice.top]
    assert probs == pytest.approx([0.665, 0.245, 0.090], abs=0.001)
    assert FakeClient().complete("x").tokens == []
