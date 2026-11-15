"""A small model client: chat, raw completion, and logprobs. FakeClient is for tests."""

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict

import httpx

DEFAULT_REPLY = "fake reply"


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class TokenChoice:
    """One generated token: its logprob, and the top candidates at that position."""

    token: str
    logprob: float
    top: list[tuple[str, float]]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    tokens: list[TokenChoice] = field(default_factory=list)


class LLMError(Exception):
    """The model could not be reached or returned an error."""


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
    ) -> LLMResponse:
        """Send messages through the model's chat template (/api/chat)."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": _options(temperature, num_ctx, num_predict, seed),
        }
        data, latency_ms = self._post("/api/chat", body, top_logprobs)
        return self._response(data["message"]["content"], data, latency_ms)

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

    def _response(self, text: str, data: dict[str, Any], latency_ms: float) -> LLMResponse:
        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=latency_ms,
            tokens=_parse_tokens(data),
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


# The guide's paper example: after "Binary search runs in O(".
PAPER_EXAMPLE = [("log", math.log(0.665)), ("n", math.log(0.245)), ("1", math.log(0.090))]


class FakeClient:
    def __init__(self, replies: Sequence[str | Exception] | None = None) -> None:
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
    ) -> LLMResponse:
        call = FakeCall(
            "chat", list(messages), None, temperature, num_ctx, num_predict, seed, top_logprobs
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
            "complete", None, prompt, temperature, num_ctx, num_predict, seed, top_logprobs
        )
        return self._reply(call)

    def _reply(self, call: FakeCall) -> LLMResponse:
        self.calls.append(call)
        if self._replies is None:
            reply: str | Exception = DEFAULT_REPLY
        elif not self._replies:
            raise AssertionError("FakeClient ran out of replies")
        else:
            reply = self._replies.pop(0)

        if isinstance(reply, Exception):
            raise reply

        tokens: list[TokenChoice] = []
        if call.top_logprobs > 0:
            tokens = [TokenChoice("log", PAPER_EXAMPLE[0][1], list(PAPER_EXAMPLE))]
        return LLMResponse(
            text=reply,
            model=self.model,
            input_tokens=100,
            output_tokens=20,
            latency_ms=1.0,
            tokens=tokens,
        )
