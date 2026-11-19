"""Tool calling with no tool API: describe the tools in plain text, ask for JSON, parse it."""

import json
import re

from ai_basics.llm import ToolCall

HANDMADE_SYSTEM = """You can use two tools.

Tool: get_course_info
  Argument: course (text, like "CS 101")
  Returns: the course title, number of students and number of sections.

Tool: calculate
  Argument: expression (text, like "84 / 3")
  Returns: the result. Supports + - * / and parentheses. Use it for every calculation.

To use a tool, reply with ONLY this JSON and nothing else:
{"tool": "<tool name>", "arguments": {"<argument name>": "<value>"}}

When you have the final answer, reply with ONLY this JSON and nothing else:
{"answer": "<your answer>"}

Never write any text outside the JSON."""

# The whole reply is one fenced block, with or without the word "json" after the backticks.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_tool_call(text: str) -> ToolCall | None:
    """Read a tool call from a model reply. Strict: anything else gives None."""
    text = text.strip()
    fenced = _FENCE.fullmatch(text)
    if fenced:
        text = fenced.group(1)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    name, arguments = data.get("tool"), data.get("arguments")
    if isinstance(name, str) and isinstance(arguments, dict):
        return ToolCall(name, arguments)
    return None
