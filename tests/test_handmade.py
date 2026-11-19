import pytest

from ai_basics.handmade import HANDMADE_SYSTEM, parse_tool_call
from ai_basics.llm import ToolCall
from ai_basics.tools import TOOLS

CALL_TEXT = '{"tool": "calculate", "arguments": {"expression": "84 / 3"}}'
CALL = ToolCall("calculate", {"expression": "84 / 3"})


def test_bare_json_is_a_tool_call():
    assert parse_tool_call(CALL_TEXT) == CALL


def test_surrounding_whitespace_is_fine():
    assert parse_tool_call(f"\n  {CALL_TEXT}  \n") == CALL


def test_json_fence_is_accepted():
    assert parse_tool_call(f"```json\n{CALL_TEXT}\n```") == CALL


def test_plain_fence_is_accepted():
    assert parse_tool_call(f"```\n{CALL_TEXT}\n```") == CALL


@pytest.mark.parametrize(
    "text",
    [
        f"Sure! {CALL_TEXT}",
        f"{CALL_TEXT} Hope that helps!",
        f"Here you go:\n```json\n{CALL_TEXT}\n```",
        '{"answer": "28"}',
        "not json at all",
        "",
        '{"tool": "calculate"}',
        '{"tool": 5, "arguments": {}}',
        '{"tool": "calculate", "arguments": "84 / 3"}',
        "[1, 2]",
        "28",
    ],
)
def test_anything_else_is_not_a_tool_call(text):
    assert parse_tool_call(text) is None


def test_system_prompt_describes_both_tools_and_both_reply_shapes():
    for word in ("get_course_info", "calculate", '"tool"', '"arguments"', '"answer"', "ONLY"):
        assert word in HANDMADE_SYSTEM


def test_a_parsed_call_can_be_run():
    call = parse_tool_call(CALL_TEXT)
    assert call is not None
    assert TOOLS[call.name](**call.arguments) == 28
