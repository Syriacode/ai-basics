import json
import math

import httpx
import pytest

from ai_basics.llm import (
    FakeClient,
    LLMError,
    OllamaClient,
    TokenChoice,
    ToolCall,
    assistant_message,
)
from ai_basics.tools import TOOL_SCHEMAS

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


# ---- tool calling ----

COURSE_CALL = ToolCall("get_course_info", {"course": "CS 101"})

TOOL_CALL_REPLY = {
    "model": "qwen2.5:3b",
    "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "get_course_info", "arguments": {"course": "CS 101"}}}
        ],
    },
    "done": True,
    "prompt_eval_count": 250,
    "eval_count": 20,
}


def reply_with_arguments(arguments):
    call = {"function": {"name": "calculate", "arguments": arguments}}
    message = {"role": "assistant", "content": "", "tool_calls": [call]}
    return {**TOOL_CALL_REPLY, "message": message}


def test_chat_without_tools_sends_no_tools_key():
    requests = []

    client_with(recording_handler(requests, CHAT_REPLY)).chat(MESSAGES)

    assert "tools" not in json.loads(requests[0].content)


def test_chat_with_tools_sends_them_and_parses_the_tool_call():
    requests = []
    client = client_with(recording_handler(requests, TOOL_CALL_REPLY))

    response = client.chat(MESSAGES, tools=TOOL_SCHEMAS)

    assert json.loads(requests[0].content)["tools"] == TOOL_SCHEMAS
    assert response.text == ""
    assert response.tool_calls == [COURSE_CALL]
    assert response.input_tokens == 250


def test_reply_without_tool_calls_gives_an_empty_list():
    assert client_with(recording_handler([], CHAT_REPLY)).chat(MESSAGES).tool_calls == []
    assert client_with(recording_handler([], GENERATE_REPLY)).complete("Hi").tool_calls == []


def test_string_arguments_are_parsed():
    # OpenAI sends the arguments as a JSON string, not an object.
    reply = reply_with_arguments('{"expression": "84 / 3"}')

    response = client_with(recording_handler([], reply)).chat(MESSAGES)

    assert response.tool_calls == [ToolCall("calculate", {"expression": "84 / 3"})]


@pytest.mark.parametrize("arguments", ["not json", "[1, 2]", '"just text"', 28])
def test_arguments_that_are_not_a_json_object_are_an_error(arguments):
    client = client_with(recording_handler([], reply_with_arguments(arguments)))

    with pytest.raises(LLMError):
        client.chat(MESSAGES)


def test_assistant_message_of_plain_text_has_no_tool_calls_key():
    response = FakeClient(["hi"]).chat(MESSAGES)

    assert assistant_message(response) == {"role": "assistant", "content": "hi"}


def test_assistant_message_keeps_tool_calls_in_ollamas_format():
    response = FakeClient([[COURSE_CALL]]).chat(MESSAGES)

    assert assistant_message(response) == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "get_course_info", "arguments": {"course": "CS 101"}}}
        ],
    }


def test_assistant_message_round_trips_through_the_client():
    message = assistant_message(FakeClient([[COURSE_CALL]]).chat(MESSAGES))
    reply = {**CHAT_REPLY, "message": message}

    response = client_with(recording_handler([], reply)).chat(MESSAGES)

    assert response.tool_calls == [COURSE_CALL]


def test_fake_client_can_script_tool_calls():
    fake = FakeClient([[COURSE_CALL], "28 per section"])

    first = fake.chat(MESSAGES, tools=TOOL_SCHEMAS)
    second = fake.chat(MESSAGES)

    assert (first.text, first.tool_calls) == ("", [COURSE_CALL])
    assert (second.text, second.tool_calls) == ("28 per section", [])
    assert fake.calls[0].tools == TOOL_SCHEMAS
    assert fake.calls[1].tools is None


def test_fake_client_records_a_copy_of_the_messages():
    fake = FakeClient()
    history = [{"role": "user", "content": "q"}]

    fake.chat(history)
    history.append({"role": "assistant", "content": "a"})
    history[0]["content"] = "changed"

    assert fake.calls[0].messages == [{"role": "user", "content": "q"}]


def test_fake_client_copy_reaches_inside_tool_calls():
    fake = FakeClient()
    call = ToolCall("get_course_info", {"course": "CS 101"})
    history = [assistant_message(FakeClient([[call]]).chat(MESSAGES))]

    fake.chat(history)
    history[0]["tool_calls"][0]["function"]["arguments"]["course"] = "CS 999"

    recorded = fake.calls[0].messages[0]["tool_calls"][0]["function"]["arguments"]
    assert recorded == {"course": "CS 101"}
