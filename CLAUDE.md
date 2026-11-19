# ai-basics

This repo is ai-basics: Small, measured experiments that show how an LLM works: tokens, sampling, context windows, system prompts, hallucination

Language: Python + Jupyter

## How to work with me

- I'm learning, so explain each change and the reasoning behind it.
- Always run the tests after changing code, and show me the output:

```bash
py -m pip install -e ".[dev,notebook]"
py -m pytest
py -m ruff check .
py -m ruff format --check .
```

- Tests never call a real model (use `FakeClient` or a mocked httpx transport). Real calls go in tests marked `@pytest.mark.live`.
- Run `py -m ruff format .` after editing Python files or notebooks.

## Plan

- `GUIDE.md` is the lab 1 guide.
- `GUIDE-lab2.md` is the lab 2 guide (tool calling and agents from scratch).

## Environment

- My machine runs Windows with Git Bash. Shell commands are POSIX bash.
