import json

import httpx

from ai_basics.agent import AGENT_SYSTEM, run_agent, run_tool, show_message
from ai_basics.llm import FakeClient, OllamaClient, ToolCall
from ai_basics.tools import TOOL_SCHEMAS, TOOLS

# The guide's paper example: CS 101 has 84 students in 3 sections, and 84 / 3 = 28.
QUESTION = "How many students per section are in CS 101?"
LOOKUP = ToolCall("get_course_info", {"course": "CS 101"})
DIVIDE = ToolCall("calculate", {"expression": "84 / 3"})
ANSWER = "CS 101 has 28 students per section."

LOOKUP_MESSAGE = {
    "role": "assistant",
    "content": "",
    "tool_calls": [{"function": {"name": "get_course_info", "arguments": {"course": "CS 101"}}}],
}


def paper_client():
    return FakeClient([[LOOKUP], [DIVIDE], ANSWER])


# ---- run_tool: errors become text, not exceptions ----


def test_run_tool_lookup_returns_json_text():
    result = run_tool(LOOKUP, TOOLS)

    assert isinstance(result, str)
    info = json.loads(result)
    assert (info["students"], info["sections"]) == (84, 3)


def test_run_tool_calculate_returns_text():
    assert run_tool(DIVIDE, TOOLS) == "28.0"


def test_run_tool_unknown_tool_is_text_listing_the_tools():
    result = run_tool(ToolCall("send_email", {"to": "class"}), TOOLS)

    assert result.startswith("error: unknown tool 'send_email'")
    assert "get_course_info" in result
    assert "calculate" in result


def test_run_tool_wrong_argument_name_is_text():
    result = run_tool(ToolCall("calculate", {"expr": "84 / 3"}), TOOLS)

    assert result.startswith("error: TypeError")


def test_run_tool_unknown_course_is_text():
    result = run_tool(ToolCall("get_course_info", {"course": "CS 999"}), TOOLS)

    assert result.startswith("error: LookupError")
    assert "CS 101" in result


# ---- run_agent: the loop ----


def test_paper_example_answers_in_three_steps():
    result = run_agent(paper_client(), QUESTION)

    assert result.answer == ANSWER
    assert result.stopped == "answer"
    assert result.steps == 3
    assert result.input_tokens == 300  # FakeClient reports 100 per call
    roles = [m["role"] for m in result.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert result.messages[0]["content"] == AGENT_SYSTEM
    assert result.messages[1]["content"] == QUESTION
    assert result.messages[5] == {"role": "tool", "tool_name": "calculate", "content": "28.0"}


def test_every_call_gets_the_tools_and_the_whole_history():
    fake = paper_client()

    run_agent(fake, QUESTION)

    assert len(fake.calls) == 3
    assert all(call.tools == TOOL_SCHEMAS for call in fake.calls)
    # The model has no memory: each call carries everything said so far.
    assert [len(call.messages) for call in fake.calls] == [2, 4, 6]


def test_model_sees_its_own_tool_call_right_before_the_result():
    fake = paper_client()

    run_agent(fake, QUESTION)

    second_call = fake.calls[1].messages
    assert second_call[-2] == LOOKUP_MESSAGE
    assert second_call[-1]["role"] == "tool"
    assert second_call[-1]["tool_name"] == "get_course_info"


def test_a_text_reply_stops_after_one_step():
    fake = FakeClient(["Hello!"])

    result = run_agent(fake, QUESTION)

    assert (result.answer, result.stopped, result.steps) == ("Hello!", "answer", 1)
    assert [m["role"] for m in result.messages] == ["system", "user", "assistant"]
    assert len(fake.calls) == 1


def test_max_steps_stops_a_model_that_only_calls_tools():
    fake = FakeClient([[LOOKUP], [LOOKUP]])

    result = run_agent(fake, QUESTION, max_steps=2)

    assert result.stopped == "max_steps"
    assert result.steps == 2
    assert len(fake.calls) == 2
    assert "(2)" in result.answer


def test_a_tool_error_reaches_the_model():
    fake = FakeClient([[ToolCall("get_course_info", {"course": "CS 999"})], "No such course."])

    result = run_agent(fake, QUESTION)

    assert result.answer == "No such course."
    last_seen = fake.calls[1].messages[-1]
    assert last_seen["role"] == "tool"
    assert last_seen["content"].startswith("error: LookupError")


def test_two_tool_calls_in_one_reply_give_two_tool_messages():
    cs230 = ToolCall("get_course_info", {"course": "CS 230"})
    fake = FakeClient([[LOOKUP, cs230], ANSWER])

    result = run_agent(fake, QUESTION)

    assert len(result.messages[2]["tool_calls"]) == 2
    first, second = result.messages[3], result.messages[4]
    assert (first["role"], second["role"]) == ("tool", "tool")
    assert "84" in first["content"]
    assert "45" in second["content"]
    assert len(fake.calls[1].messages) == 5


def test_verbose_prints_each_step(capsys):
    run_agent(paper_client(), QUESTION, verbose=True)

    out = capsys.readouterr().out
    assert "step 1" in out
    assert '[assistant wants] get_course_info({"course": "CS 101"})' in out
    assert "[tool → calculate] 28.0" in out


def test_quiet_by_default(capsys):
    run_agent(paper_client(), QUESTION)

    assert capsys.readouterr().out == ""


# ---- the same loop against a real OllamaClient, with the network replaced ----


def ollama_reply(**message):
    return {
        "model": "qwen2.5:3b",
        "message": {"role": "assistant", **message},
        "done": True,
        "prompt_eval_count": 100,
        "eval_count": 10,
    }


def tool_call_reply(name, arguments):
    return ollama_reply(
        content="", tool_calls=[{"function": {"name": name, "arguments": arguments}}]
    )


def test_loop_works_with_ollama_client_on_a_mock_transport():
    replies = iter(
        [
            tool_call_reply("get_course_info", {"course": "CS 101"}),
            tool_call_reply("calculate", {"expression": "84 / 3"}),
            ollama_reply(content=ANSWER),
        ]
    )
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=next(replies))

    http = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(handler))

    result = run_agent(OllamaClient(http=http), QUESTION)

    assert (result.answer, result.stopped, result.steps) == (ANSWER, "answer", 3)
    assert result.input_tokens == 300
    assert len(bodies) == 3
    assert all(body["tools"] == TOOL_SCHEMAS for body in bodies)
    assert [len(body["messages"]) for body in bodies] == [2, 4, 6]
    last_sent = bodies[1]["messages"][-1]
    assert (last_sent["role"], last_sent["tool_name"]) == ("tool", "get_course_info")
    assert json.loads(last_sent["content"])["students"] == 84


# ---- show_message: one line per message ----


def test_show_message_tool_result(capsys):
    show_message({"role": "tool", "tool_name": "calculate", "content": "28.0"})

    assert capsys.readouterr().out == "  [tool → calculate] 28.0\n"


def test_show_message_tool_call(capsys):
    show_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "calculate", "arguments": {"expression": "84 / 3"}}}
            ],
        }
    )

    assert capsys.readouterr().out == '  [assistant wants] calculate({"expression": "84 / 3"})\n'


def test_show_message_plain_text(capsys):
    show_message({"role": "user", "content": "hi"})

    assert capsys.readouterr().out == "  [user] hi\n"
