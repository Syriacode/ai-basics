# ai-basics

Small, measured experiments that show how an LLM works: tokens, sampling, context windows, system prompts,
hallucination, and then tool calling and agents built from scratch.

Part of my [learning roadmap](https://github.com/Syriacode/Mini-Blockchain/blob/main/ROADMAP.md).

| | |
|---|---|
| Language | Python + Jupyter |
| Claude Code technique practiced | Lab 1: Claude as a tutor (custom output style). Lab 2: watching Claude Code's own agent loop (transcript viewer, `claude -p --output-format stream-json`) |

## Run it

```bash
py -m pip install -e ".[dev,notebook]"
py -m pytest
py -m ruff check .
py -m ruff format --check .
```

Live tests call the real model, so they need Ollama running with `qwen2.5:3b`. They are skipped by default:

```bash
py -m pytest -m live
```

Open a notebook and run the cells with Shift+Enter:

```bash
py -m notebook notebooks/lab1_how_llms_work.ipynb
py -m notebook notebooks/lab2_tool_calling_and_agents.ipynb
```

To run the notebooks with no model (this is what CI does), set `AI_BASICS_PROVIDER=fake`. The setup cell then uses
`FakeClient` instead of Ollama:

```bash
AI_BASICS_PROVIDER=fake MPLBACKEND=Agg py -m jupyter nbconvert --to notebook --execute --output-dir /tmp \
  notebooks/lab1_how_llms_work.ipynb notebooks/lab2_tool_calling_and_agents.ipynb
```

## Lab 1: how an LLM works

| Part | Question | Experiment | My result |
|---|---|---|---|
| A | What does the model read? | token count and token pieces for 5 texts | The English sentence (23 characters) was 7 tokens (3.29 characters per token). `unbelievably` was 3 tokens (un, belie, vably). The code line was 7 tokens (4.43). The Spanish sentence (32 characters) took 12 tokens (2.67), more than the English one. `1234567` was 7 tokens, one per digit (1.00). |
| B | How does it write? | top 5 next-token guesses | After "The capital of France is", `' Paris'` had 0.616. After "Binary search runs in O(", `' log'` had 0.510. After "My favorite programming language is", `' Python'` had 0.663. |
| C | What does temperature do? | the same question 5 times | At T=0 there was 1 different answer (`TortoiseTom` five times). At T=1 there were 5 different answers. |
| D | What is a context window? | a secret in the first of 120 notes | With `num_ctx` 4096 the model saw 2,587 input tokens and found `BLUE-42`. With `num_ctx` 1024 it saw 514 input tokens, did not find it, and answered `'120'`. There was no error. |
| E | Are system messages special? | the same rule in the system message, the user message, and nowhere | Refusals out of 5 off-topic questions: rule in the system message 3/5, in the user message 5/5, nowhere 0/5. |
| F | Why do models make things up? | 3 invented things, asked plainly and with "I don't know" allowed | Even the plain answers said they had no record of the invented person, paper and algorithm. Average token probability was 0.79, 0.77 and 0.75. With "I don't know" allowed, it said so 3 times out of 3. |

These numbers come from one run on my machine (`qwen2.5:3b` on a CPU). Three results surprised me: ` Paris` was
only 0.616 after "The capital of France is"; the 1024-token run saw 514 input tokens, about half of `num_ctx` (I
have not worked out why); and the rule in the user message (5/5) beat the same rule in the system message (3/5).

## Lab 2: tool calling and agents

| Part | Question | Experiment | My result |
|---|---|---|---|
| A | Can a model call a tool using only text? | a JSON-only system prompt, the same question 5 times at T=1.0 | 5 of 5 replies parsed as a `get_course_info` call for `CS 101`, and my code ran the tool. |
| B | What does the tool API change? | the same question with and without `tools=` | Without tools the model could not answer (42 input tokens). With tools it asked for `get_course_info({"course": "CS 101"})` (265 input tokens). The tool descriptions cost 223 tokens. |
| C | What does one agent run look like? | "How many students per section are in CS 101?" with `verbose=True` | 2 model calls (284 + 356 = 640 input tokens), one tool call, stopped with `answer`: 28 students per section. |
| D1 | What if the step limit is too low? | `max_steps=1` | One tool call, then `stopped: max_steps` and "I hit the step limit (1) before finding an answer." The model never read the lookup result. |
| D2 | What if a tool fails? | "How many students per section are in CS 999?" | The tool returned `error: LookupError: unknown course 'CS 999'; known courses: CS 101, CS 230, CS 350` as text. The model told the user the course was not found. |
| D3 | What if no tool is needed? | "What does CPU stand for?" | 1 step (277 input tokens), no tool call, `stopped: answer`. |
| D4 | Do tools make the model more reliable? | 3 courses and a $37 textbook, with all tools and with only `get_course_info` | Both runs looked up all 3 courses in one step. With all tools the final reply was empty (it never called `calculate`). With only the lookup it answered $1,173. The true total is 6,105. |

These numbers come from one run on my machine (`qwen2.5:3b` on a CPU), so another run may differ. D4 surprised me:
the model with the calculator available gave no answer, and the one without it gave a wrong one.

## Design decisions

### Lab 1

- **Raw completion for next-token experiments, chat for the rest.** `/api/generate` with `raw: true` skips the chat
  template, so the model just continues the text. Chat is used where a reply to a question is the point.
- **Logprobs are the window into the model.** For each token the model reports the top candidates as log
  probabilities, and `e^logprob` turns them back into probabilities.
- **Softmax and sampling are written by hand** and tested against the paper numbers (0.665, 0.245, 0.090, and the
  rows for temperature 0.5 and 2), so the same numbers appear on paper and in code.
- **`FakeClient` returns the paper example** when logprobs are asked for, so CI can run the whole notebook with no
  model.
- **A Tutor output style is kept in the repo** (`.claude/output-styles/tutor.md`), so it is versioned and reusable.

### Lab 2

- **Tool calling by hand first, then the API.** Part A (`ai_basics/handmade.py`) parses JSON out of the model's
  text, and Part B uses the chat API's `tools=` field, so the mechanism is visible before the API hides it.
- **A ~40-line agent loop with no framework.** `run_agent` in `ai_basics/agent.py` is a plain loop: call the model,
  run the tools it asked for, append the results, repeat.
- **Tool errors are returned to the model as text.** The model can read the message and recover, as D2 shows.
- **Two stopping conditions: `answer` and `max_steps`.** The model answers without asking for a tool, or the step
  limit stops the loop.
- **`calculate()` walks the `ast` and never evals model output.** `ai_basics/tools.py` accepts only numbers,
  `+ - * /` and unary signs. The model chooses the arguments, so they are untrusted input.
- **`FakeClient` scripts tool calls**, so CI can run both notebooks with no model.

## Known limitations

### Lab 1

- One small model (3B) on a CPU.
- Each experiment uses 3–5 samples, so the numbers are illustrations, not measurements.
- Token counts are read from `prompt_eval_count`, because Ollama has no tokenize endpoint.
- The "I don't know" check is a phrase match, so it can miss other ways of admitting doubt.
- The truncation result depends on the Ollama version. Mine showed 514 input tokens at `num_ctx` 1024.

### Lab 2

- One small model on a CPU.
- No human approval, no time limit and no token budget. Only `max_steps` limits the loop.
- Tool results are matched to calls by name, not by id, so two calls to the same tool in one step rely on order.
- The whole history is sent again at every step, with no caching or trimming, so input tokens grow each step.
- The course data is invented.
- An empty reply from the model counts as an "answer", so the loop can stop with nothing useful (D4, all tools).
