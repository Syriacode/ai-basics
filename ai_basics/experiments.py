"""Small experiment functions that the lab notebook calls."""

from collections.abc import Sequence

from ai_basics.llm import LLMClient, Message, TokenChoice

ECHO_SYSTEM = "Repeat the user's text exactly. Output nothing else."


def show_tokens(tokens: Sequence[TokenChoice]) -> str:
    """Join token strings with "|", making spaces ("·") and newlines ("\\n") visible."""
    return "|".join(t.token.replace(" ", "·").replace("\n", "\\n") for t in tokens)


def count_tokens(client: LLMClient, text: str) -> int:
    """How many tokens the model read: send the text raw and write just one token."""
    return client.complete(text, num_predict=1).input_tokens


def echo_tokens(client: LLMClient, text: str) -> list[TokenChoice]:
    """Ask the model to repeat the text; each logprob entry in the reply is one token."""
    messages: list[Message] = [
        {"role": "system", "content": ECHO_SYSTEM},
        {"role": "user", "content": text},
    ]
    return client.chat(messages, top_logprobs=1).tokens
