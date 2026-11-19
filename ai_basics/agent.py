"""A tool-using agent from scratch: ask, run the tools the model asks for, repeat."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from ai_basics.llm import LLMClient, Message, ToolCall, assistant_message  # noqa: F401
from ai_basics.tools import TOOL_SCHEMAS, TOOLS

# assistant_message is not used yet. run_agent needs it in Step 5, so the noqa keeps ruff
# from removing the import before you type that code.

AGENT_SYSTEM = (
    "You are a course assistant. Use the tools to look up facts and to do arithmetic; "
    "never guess a number. When you have the answer, reply in one short sentence."
)


@dataclass(frozen=True)
class AgentResult:
    answer: str
    stopped: Literal["answer", "max_steps"]
    steps: int
    input_tokens: int
    messages: list[Message]


def show_message(message: Message) -> None:
    """Print one message as one short line, indented two spaces (one line per tool call)."""
    if message["role"] == "tool":
        print(f"  [tool → {message.get('tool_name', '?')}] {message['content']}")
        return

    tool_calls = message.get("tool_calls", [])
    for call in tool_calls:
        function = call["function"]
        print(f"  [assistant wants] {function['name']}({json.dumps(function['arguments'])})")
    if not tool_calls:
        print(f"  [{message['role']}] {message['content']}")


def run_tool(call: ToolCall, tools: dict[str, Callable[..., Any]]) -> str:
    """Run one tool call and return the result as text. Errors become text too, for the model."""
    if call.name not in tools:
        return f"error: unknown tool {call.name!r}. Available tools: {', '.join(tools)}"
    try:
        result = tools[call.name](**call.arguments)
    except Exception as exc:  # bad arguments, unknown course, division by zero...
        return f"error: {type(exc).__name__}: {exc}"
    return json.dumps(result)


def run_agent(
    client: LLMClient,
    question: str,
    *,
    tools: dict[str, Callable[..., Any]] = TOOLS,
    schemas: list[dict[str, Any]] = TOOL_SCHEMAS,
    max_steps: int = 5,
    verbose: bool = False,
) -> AgentResult:
    """Ask the model, run the tools it asks for, send the results back, repeat until it answers."""
    raise NotImplementedError
