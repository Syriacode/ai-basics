"""A small model client: chat, raw completion, and logprobs. FakeClient is for tests."""

import copy
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, Protocol, TypedDict

import httpx

DEFAULT_REPLY = "fake reply"


class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    # Only on an assistant message that asked for tools (Ollama's format).
    tool_calls: NotRequired[list[dict[str, Any]]]
    # Only on a tool message: which tool produced this result.
    tool_name: NotRequired[str]


@dataclass(frozen=True)
class TokenChoice:
    """One generated token: its logprob, and the top candidates at that position."""

    token: str
    logprob: float
    top: list[tuple[str, float]]


@dataclass(frozen=True)
class ToolCall:
    """The model asked to run the tool `name` with these keyword arguments."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    tokens: list[TokenChoice] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMError(Exception):
    """The model could not be reached or returned an error."""


def assistant_message(response: LLMResponse) -> Message:
    """The reply as a history message. Tool calls stay in it, in Ollama's format.

    The model must see its own call before the tool's result, so an agent adds this
    message to the history before it adds the result.
    """
    message: Message = {"role": "assistant", "content": response.text}
    if response.tool_calls:
        message["tool_calls"] = [
            {"function": {"name": call.name, "arguments": dict(call.arguments)}}
            for call in response.tool_calls
        ]
    return message


class LLMClient(Protocol):
    model: str

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        num_ctx: int = 4096,
        num_predict: int | None = None,
        seed: int | None = None,
        top_logprobs: int = 0,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        num_ctx: int = 4096,
        num_predict: int | None = None,
        seed: int | None = None,
        top_logprobs: int = 0,
    ) -> LLMResponse: ...


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        http: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self._http = http or httpx.Client(base_url=base_url, timeout=timeout)

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        num_ctx: int = 4096,
        num_predict: int | None = None,
        seed: int | None = None,
        top_logprobs: int = 0,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send messages through the model's chat template (/api/chat)."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": _options(temperature, num_ctx, num_predict, seed),
        }
        if tools is not None:
            body["tools"] = list(tools)
        data, latency_ms = self._post("/api/chat", body, top_logprobs)
        message = data["message"]
        return self._response(
            message["content"], data, latency_ms, tool_calls=_parse_tool_calls(message)
        )

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        num_ctx: int = 4096,
        num_predict: int | None = None,
        seed: int | None = None,
        top_logprobs: int = 0,
    ) -> LLMResponse:
        """Continue raw text with no chat template (/api/generate with raw=true)."""
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": _options(temperature, num_ctx, num_predict, seed),
        }
        data, latency_ms = self._post("/api/generate", body, top_logprobs)
        return self._response(data["response"], data, latency_ms)

    def _post(
        self, path: str, body: dict[str, Any], top_logprobs: int
    ) -> tuple[dict[str, Any], float]:
        if top_logprobs > 0:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs

        start = time.perf_counter()
        try:
            response = self._http.post(path, json=body)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000
        return response.json(), latency_ms

    def _response(
        self,
        text: str,
        data: dict[str, Any],
        latency_ms: float,
        tool_calls: list[ToolCall] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=latency_ms,
            tokens=_parse_tokens(data),
            tool_calls=tool_calls or [],
        )


def _options(
    temperature: float, num_ctx: int, num_predict: int | None, seed: int | None
) -> dict[str, Any]:
    options: dict[str, Any] = {"temperature": temperature, "num_ctx": num_ctx}
    if num_predict is not None:
        options["num_predict"] = num_predict
    if seed is not None:
        options["seed"] = seed
    return options


def _parse_tokens(data: dict[str, Any]) -> list[TokenChoice]:
    """Read Ollama's "logprobs" list. A missing field (old Ollama) gives an empty list."""
    return [
        TokenChoice(
            token=entry["token"],
            logprob=entry["logprob"],
            top=[(t["token"], t["logprob"]) for t in entry.get("top_logprobs") or []],
        )
        for entry in data.get("logprobs") or []
    ]


def _parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    """Read message["tool_calls"]. Arguments are a JSON object, or a JSON string (OpenAI)."""
    calls = []
    for entry in message.get("tool_calls") or []:
        function = entry["function"]
        name = function["name"]
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise LLMError(f"tool call {name!r} has arguments that are not JSON") from exc
        if not isinstance(arguments, dict):
            raise LLMError(f"tool call {name!r} has arguments that are not a JSON object")
        calls.append(ToolCall(name, arguments))
    return calls


@dataclass(frozen=True)
class FakeCall:
    """What a FakeClient was asked, so tests can check it."""

    method: Literal["chat", "complete"]
    messages: list[Message] | None
    prompt: str | None
    temperature: float
    num_ctx: int
    num_predict: int | None
    seed: int | None
    top_logprobs: int
    tools: list[dict[str, Any]] | None = None


# The guide's paper example: after "Binary search runs in O(".
PAPER_EXAMPLE = [("log", math.log(0.665)), ("n", math.log(0.245)), ("1", math.log(0.090))]


class FakeClient:
    def __init__(self, replies: Sequence[str | list[ToolCall] | Exception] | None = None) -> None:
        self.model = "fake"
        self.calls: list[FakeCall] = []
        self._replies = None if replies is None else list(replies)

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        num_ctx: int = 4096,
        num_predict: int | None = None,
        seed: int | None = None,
        top_logprobs: int = 0,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        call = FakeCall(
            method="chat",
            # A deep copy: the caller keeps changing its list, and we want what the model saw.
            messages=copy.deepcopy(list(messages)),
            prompt=None,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            seed=seed,
            top_logprobs=top_logprobs,
            tools=None if tools is None else copy.deepcopy(list(tools)),
        )
        return self._reply(call)

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        num_ctx: int = 4096,
        num_predict: int | None = None,
        seed: int | None = None,
        top_logprobs: int = 0,
    ) -> LLMResponse:
        call = FakeCall(
            method="complete",
            messages=None,
            prompt=prompt,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            seed=seed,
            top_logprobs=top_logprobs,
        )
        return self._reply(call)

    def _reply(self, call: FakeCall) -> LLMResponse:
        self.calls.append(call)
        if self._replies is None:
            reply: str | list[ToolCall] | Exception = DEFAULT_REPLY
        elif not self._replies:
            raise AssertionError("FakeClient ran out of replies")
        else:
            reply = self._replies.pop(0)

        if isinstance(reply, Exception):
            raise reply

        # A list reply is "the model asked for these tools": no text, only tool calls.
        text, tool_calls = ("", list(reply)) if isinstance(reply, list) else (reply, [])

        tokens: list[TokenChoice] = []
        if call.top_logprobs > 0:
            tokens = [TokenChoice("log", PAPER_EXAMPLE[0][1], list(PAPER_EXAMPLE))]
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=100,
            output_tokens=20,
            latency_ms=1.0,
            tokens=tokens,
            tool_calls=tool_calls,
        )
