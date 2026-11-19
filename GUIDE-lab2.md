# Phase 2 Wed — AI basics lab 2: tool calling and agents from scratch (step-by-step guide)

> Read the **Project overview** first, on its own, before you open a terminal. It teaches today's ideas with small
> examples you can work out on paper. The steps start at Step 0.
>
> Natural stop after **Step 3** (about 1 hour). Steps 4–9 next time. The **Check your understanding** questions at
> the end are for the reading day.

---

## Project overview

### 1. What you've built so far (in plain words)

- **Lab 1 (Sat, `ai-basics`):** you opened the black box. A model reads **tokens**, scores every possible next
  token, turns the scores into probabilities with **softmax**, picks one (temperature decides how adventurous the
  pick is), appends it, and runs again. Chat messages (system, user, assistant) are glued into **one long token
  stream** by a **chat template**. Anything that doesn't fit in the **context window** is silently cut.
- **CourseMate API (Mon):** the same `LLMClient` idea behind a web API, with streaming, sessions, and three
  backends (Ollama, OpenAI format, Anthropic format) you can swap with a setting.

Every one of those was **one question in, one answer out**. The model could only use what you put in the prompt.
Today it gets **hands**: it can ask your code to look something up or calculate something, read the result, and
decide what to do next. That's all an **agent** is, and you'll build one in about 40 lines with no framework.

| Part | Question | What you'll see |
|---|---|---|
| A | What is "tool calling", really? | the model writes JSON in plain text; **your code** parses it and runs the tool |
| B | What does a tool API add? | the same thing, built into the chat template and the server's parser |
| C | What is an agent? | a loop: ask the model, run its tools, send results back, repeat, with every message printed |
| D | How does an agent stop (or fail to)? | a step limit, a tool error, a question with no tool, and a missing calculator |

This is the foundation for the next three projects: the **CourseMate agent** (Fri, Claude Agent SDK), the
**MCP server** (Sun, tools served to other apps), and **LangGraph** (Phase 5). Today you learn what those
frameworks do for you, so later you know what they're hiding.

### 2. The AI basics you need today (work these on paper)

#### 2a. Tool calling: the model only writes text; your code does the work

A language model can't open a file, run a calculator or call an API. It can **only produce tokens**. "Tool calling"
(also called **function calling**) is a convention:

1. You tell the model which tools exist: a name, what it does, and what arguments it takes.
2. When the model wants a tool, it **writes text in an agreed format**, usually JSON: "call `get_course_info` with
   `course = "CS 101"`".
3. **Your code** reads that text, runs the real Python function, and sends the result back as a new message.
4. The model reads the result and either asks for another tool or answers.

Paper example. Our invented course data:

```text
 course   title                  students  sections
 CS 101   Intro to Programming   84        3
 CS 230   Data Structures        45        2
 CS 350   Operating Systems      36        1
```

Question: *"How many students per section are in CS 101?"* The model doesn't know these numbers (they're
invented), so a good model asks:

```text
 model writes:     {"tool": "get_course_info", "arguments": {"course": "CS 101"}}
 your code runs:   get_course_info(course="CS 101")
 your code sends:  {"course": "CS 101", "title": "Intro to Programming", "students": 84, "sections": 3}
 model writes:     {"tool": "calculate", "arguments": {"expression": "84 / 3"}}
 your code runs:   calculate(expression="84 / 3")   →  28.0
 model writes:     "CS 101 has 28 students per section."
```

**The key sentence for interviews:** *the model never runs anything. It only proposes a call; your code decides
whether and how to run it.* That's also why tools are a security boundary (section 2g).

#### 2b. Describing a tool: JSON Schema

How does the model know what arguments `get_course_info` takes? You send a description in **JSON Schema**, a
standard way to describe the shape of JSON data ("an object with a string field `course`, which is required"):

```json
{
  "type": "function",
  "function": {
    "name": "get_course_info",
    "description": "Look up a course by its code, e.g. 'CS 101'. Returns the course title, number of students and number of sections.",
    "parameters": {
      "type": "object",
      "properties": {
        "course": {"type": "string", "description": "The course code, e.g. 'CS 101'"}
      },
      "required": ["course"]
    }
  }
}
```

Three parts, and each one matters:

- **`name`:** what the model writes to call it. Your code looks the function up by this name.
- **`description`:** the model's only clue about **when** to use the tool. A vague description ("gets info") gives a
  model that calls the wrong tool. It's a prompt, so write it like one.
- **`parameters`:** the argument names and types. The model fills them in. It can still get them wrong (a missing
  argument, `"cs101"` instead of `"CS 101"`), so your code must check.

You already know this idea from Pydantic in the triage CLI: Pydantic models **produce** JSON Schema
(`Model.model_json_schema()`). Frameworks use exactly that to describe tools for you.

#### 2c. How a tool API works underneath (the "underlying technology")

In Part A you'll describe the tools in plain English and ask for JSON. That works, but small models sometimes wrap
the JSON in ```` ```json ```` fences, add "Sure!" before it, or answer directly. Real tool APIs make this reliable
with three pieces that work together:

1. **Training.** Model makers fine-tune the model on many example conversations where a tool is called in one exact
   format. Qwen 2.5 (your model) was trained to write tool calls inside `<tool_call>` ... `</tool_call>` tags.
2. **The chat template.** When you pass `tools=[...]` to Ollama, the chat template (Lab 1, section 2e) **writes your
   JSON Schemas into the system message as text**, with instructions in the format the model was trained on.
3. **A parser on the server.** When the model's output contains `<tool_call>{...}</tool_call>`, Ollama cuts it out
   of the text, parses the JSON, and returns it as a separate `tool_calls` field. You never see the tags.

Roughly what `qwen2.5:3b` really reads (you'll print the real template in Step 1):

```text
 <|im_start|>system
 You are a course assistant. ...

 # Tools
 You may call one or more functions to assist with the user query.
 You are provided with function signatures within <tools></tools> XML tags:
 <tools>
 {"type": "function", "function": {"name": "get_course_info", "description": ..., "parameters": ...}}
 {"type": "function", "function": {"name": "calculate", ...}}
 </tools>
 For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags
 <|im_end|>
 <|im_start|>user
 How many students per section are in CS 101?<|im_end|>
 <|im_start|>assistant
 <tool_call>
 {"name": "get_course_info", "arguments": {"course": "CS 101"}}
 </tool_call><|im_end|>
 <|im_start|>user
 <tool_response>
 {"course": "CS 101", "title": "Intro to Programming", "students": 84, "sections": 3}
 </tool_response><|im_end|>
 <|im_start|>assistant
```

So a "tool API" is **still next-token prediction**: tools become tokens in, a tool call is tokens out. Two
consequences you'll measure:

- **Tools cost tokens.** Every tool description is in every request. Ten tools with long descriptions can be
  1,000+ input tokens before the user says anything.
- **It can still fail.** A small model can write a broken call, or call a tool when it shouldn't. The API makes it
  more reliable, not guaranteed.

#### 2d. The agent loop

An **agent** is a model in a loop that decides its own next step from what the tools return. The whole idea, in
pseudocode:

```text
 messages = [system, user question]
 repeat at most max_steps times:
     reply = model(messages, tools)
     add reply to messages
     if reply has no tool calls:  return reply.text          ← the model is done
     for each tool call in reply:
         result = run the tool (errors become text)
         add result to messages as a "tool" message
 return "I hit the step limit"                               ← the safety net
```

Trace it on paper for the CS 101 question. Each row is one message in the list:

```text
 #  role        content                                                    added by
 0  system      "You are a course assistant. Use the tools..."             you, at the start
 1  user        "How many students per section are in CS 101?"             you, at the start
 ── step 1: model call 1 sees messages 0–1
 2  assistant   tool_calls: get_course_info(course="CS 101")               the model
 3  tool        {"course": "CS 101", ..., "students": 84, "sections": 3}   your code
 ── step 2: model call 2 sees messages 0–3
 4  assistant   tool_calls: calculate(expression="84 / 3")                 the model
 5  tool        28.0                                                       your code
 ── step 3: model call 3 sees messages 0–5
 6  assistant   "CS 101 has 28 students per section."   (no tool calls → stop)
```

Count: **3 model calls, 2 tool runs, 7 messages.** You'll check these exact numbers in the tests.

Two details people get wrong:

- **The model has no memory between calls.** It "remembers" the lookup only because you send messages 0–3 again
  in step 2. The message list **is** the agent's memory.
- **The assistant's tool-call message (row 2) must go into the list before the result (row 3).** Without it, the
  model sees a result for a call it never made.

Why "from scratch" matters: frameworks (LangGraph, the Claude Agent SDK, OpenAI's Agents SDK) wrap this loop and add
retries, state, streaming and approvals. The loop itself is this small. This pattern of thinking, acting with a
tool, and observing the result was popularized by the **ReAct** paper (2022: "Reason + Act").

#### 2e. Stopping conditions: how the loop ends

| Stop | When | In your code |
|---|---|---|
| **Final answer** | the model replies with text and no tool calls | `if not response.tool_calls: return ...` |
| **Step limit** | the model keeps calling tools (stuck in a loop, or the task is too big) | `for step in range(1, max_steps + 1)` |

And one thing that is **not** a stop: a **tool error**. If the model asks for `CS 999`, the lookup fails. Instead
of crashing, your code sends the error back as text (`error: LookupError: no course 'CS 999'; known courses: CS
101, CS 230, CS 350`). The model reads it and can fix its call or tell the user. Real agents add more stops: a time
limit, a token or money budget, and "ask a human" (Friday's project).

#### 2f. Why agents cost more: the whole history goes back every step

Paper example. Say the system message + tool descriptions + question = **300 tokens**, each tool call adds about
**25** tokens, the lookup result **30**, and the calculator result **5**:

```text
 step 1 input:  300
 step 2 input:  300 + 25 (call) + 30 (result)  = 355
 step 3 input:  355 + 25 (call) +  5 (result)  = 385
                                         total = 1,040 input tokens   (vs. 300 for one plain call)
```

Three steps cost about **3.5×** one call, and it grows faster with more steps, because every step re-sends
everything before it. On a hosted API that's money; on your CPU it's seconds. You'll print the real numbers in
Part C. (Hosted APIs reduce this with **prompt caching**: the repeated beginning of the prompt is cheaper the second
time. That's in the Nov 24 observability project.)

#### 2g. Safety: the model chooses the arguments, so treat them as untrusted input

The model writes the arguments, and the model's input can include text from users, web pages or documents. So the
arguments are **untrusted**, like a form field on a website. Our calculator is the classic example:

```python
eval("84 / 3")  # 28.0 — looks fine
eval("__import__('os').system('rm -rf ~')")  # runs a shell command. Never do this.
```

So `calculate()` uses Python's **`ast`** module (abstract syntax tree) instead. `ast.parse("84 / 3")` turns the
text into a tree **without running it**:

```text
 "84 / 3"  →  BinOp(left=Constant(84), op=Div(), right=Constant(3))     only numbers and + - * /  → allowed → 28.0
 "__import__('os')..."  →  Call(func=Attribute(...))                     a function call          → ValueError
```

The code walks the tree and allows only numbers and `+ - * /`. Anything else is refused. The same rule holds for
every tool: check the arguments, and don't give the model a tool that can do damage (send email, delete files)
without a human approving it. You'll attack this on purpose on Nov 27 (prompt injection lab).

#### 2h. Workflow or agent?

- A **workflow** is a fixed path **you** wrote: your triage CLI always does "classify → validate → log". The model
  fills in a step, but it doesn't choose the steps.
- An **agent** lets **the model** choose the next step from what it has seen.

Agents are more flexible and harder to test, slower and more expensive. A good interview answer: use a workflow
when you know the steps; use an agent when the steps depend on what you find.

### 3. What we'll build (the parts and how they fit)

```text
 notebooks/lab2_tool_calling_and_agents.ipynb    Parts A–D: you run cells, read every message, write "What I learned"
          │ calls
          ▼
 ai_basics/agent.py      run_agent()  ← YOU type: the loop       (Step 5)
          │              run_tool()   ← YOU type: name → function (Step 4)
          │              show_message(): prints one message per line
          ├──────────────────────────────┐
          ▼                              ▼
 ai_basics/tools.py                ai_basics/llm.py
 get_course_info(), calculate()    chat(messages, tools=...) → LLMResponse(text, tool_calls=[ToolCall...])
 TOOLS: name → function            assistant_message(): the model's reply as a history message
 TOOL_SCHEMAS: JSON Schemas        FakeClient: plays back scripted text or tool calls (tests and CI)
          ▲
 ai_basics/handmade.py   Part A only: tools described in plain text, parse_tool_call() reads the JSON
          │
          ▼
 Ollama on your PC  ──►  qwen2.5:3b   (tools go in through its chat template)
```

| New or changed file | What it is |
|---|---|
| `ai_basics/tools.py` | the two tools, the course data, and their JSON Schemas |
| `ai_basics/handmade.py` | Part A: tool calling with no tool API |
| `ai_basics/llm.py` (changed) | `chat()` takes `tools`, the reply has `tool_calls`; `FakeClient` can script tool calls |
| `ai_basics/agent.py` | the agent loop: **you type `run_tool` and `run_agent`** |
| `notebooks/lab2_tool_calling_and_agents.ipynb` | the lab notebook |
| `tests/test_tools.py`, `test_handmade.py`, `test_agent.py` | tests with the paper numbers (84, 3, 28) |

Same repo as lab 1, on a branch called `lab2`, merged with a pull request at the end.

### 4. How we'll test it

1. **Your two functions** are tested with the **paper trace from section 2d**: a `FakeClient` scripted to ask for
   `get_course_info("CS 101")`, then `calculate("84 / 3")`, then answer. The tests check 3 steps, the 7 roles in
   order, `"students": 84` in the first result and `28.0` in the second.
2. **The stops** each get a test: a text answer stops after 1 step, a model that never answers stops at
   `max_steps`, and a tool error goes back to the model instead of crashing.
3. **The tools** are tested on their own, including the attack: `calculate("__import__('os')...")` must raise.
4. **The client** is tested against a mocked Ollama (`httpx.MockTransport`): `tools` is sent, `tool_calls` is read.
5. **One live test** (`py -m pytest -m live`) runs the agent on the real model and checks the answer has `28`.
6. **CI** runs lint, the tests, and **both notebooks** top to bottom with the fake model.

### 5. New words (short definitions)

- **Tool / function calling:** the model writes a structured request ("call X with these arguments"); your code
  runs it and sends back the result. **Tool call:** one such request. **Tool result:** what your code sends back.
- **JSON Schema:** a standard JSON format that describes the shape of other JSON (fields, types, required).
- **Agent:** a model in a loop that chooses its next action from tool results. **Agent loop:** that loop.
  **Step:** one model call in the loop. **Stopping condition:** what ends the loop.
- **Workflow:** a fixed sequence of steps written in code, with the model doing parts of it.
- **Tool registry:** the dictionary from tool name to Python function (`TOOLS`). **Dispatch:** looking up the name
  and calling the function (`run_tool`).
- **Parallel tool calls:** several tool calls in one model reply (you'll see this in Part D).
- **`ast` (abstract syntax tree):** Python's parsed form of code, a tree you can inspect without running it.
- **ReAct:** "reason + act", the pattern of letting a model alternate between thinking, calling a tool and reading
  the result.
- **Harness:** the code around a model that runs the loop, the tools and the permissions. Claude Code is a harness.
- **Transcript (Claude Code):** the full record of a session: your messages, Claude's messages, every tool call and
  every tool result.

### 6. Today's Claude Code technique: watch Claude Code's own agent loop

**Claude Code is an agent built exactly like the one you're writing**, just bigger: a model, a list of messages, a
dozen or so tools (Read, Edit, Write, Bash, Glob, Grep, ...), and a loop. When you ask it something, it calls tools
and reads the results until it can answer. Normally you see a short summary. Press **`Ctrl+O`** to open the
**transcript viewer**: the detailed tool calls and results, with a timestamp and the model used on each assistant
message. Press `Esc` (or `q`) to go back.

In Step 8 you'll go one level deeper: run Claude Code headless with `--output-format stream-json --verbose`, which
prints **every message as one JSON line**, and compare it with your own loop:

| Your agent (today) | Claude Code |
|---|---|
| `run_agent()`: a `for` loop | its agent loop, the same shape |
| 2 tools in `TOOL_SCHEMAS` | many tools (Read, Edit, Bash, Glob, ...), plus any MCP servers you add |
| `run_tool()` looks the name up | runs the tool, but first checks **permissions** (your allow/deny rules, or asks you) |
| tool errors go back as text | a failed command's output goes back to the model, the same way |
| `max_steps` | it runs until the model answers, or you press `Esc` |
| messages with role `"tool"` | Anthropic's format: `tool_use` blocks from the model, `tool_result` blocks back |

The permission prompt you've been clicking since Day 0 is **human-in-the-loop**: a person approves a tool call
before it runs. That's what you'll add to CourseMate on Friday.

### 7. Talking to hosted APIs (interviewers will ask)

The idea is the same everywhere; the field names differ:

| | Ollama (today) | OpenAI Chat Completions | Anthropic Messages |
|---|---|---|---|
| tool description | `tools: [{type: "function", function: {name, description, parameters}}]` | same as Ollama | `tools: [{name, description, input_schema}]` |
| model's call | `message.tool_calls[].function` with `arguments` as a **JSON object** | `message.tool_calls[]` with an `id`, and `arguments` as a **JSON string** | a content block `{"type": "tool_use", "id", "name", "input"}`, and `stop_reason: "tool_use"` |
| your result | `{"role": "tool", "tool_name": ..., "content": ...}` | `{"role": "tool", "tool_call_id": ..., "content": ...}` | a **user** message with `{"type": "tool_result", "tool_use_id", "content"}` |

Why the **`id`**? When the model makes several calls at once, the id says which result belongs to which call. Ollama
matches results by tool name and order instead. Your client accepts `arguments` as a JSON string too, so an OpenAI
backend wouldn't need a rewrite (that's the `LLMClient` idea again).

### 8. Time and cost

About **2 hours** of hands-on time, plus reading. Natural stop after **Step 3**. **Cost: $0**: `qwen2.5:3b` runs
locally, and Step 8 uses your Claude subscription the same way any Claude Code session does. An agent run makes
2–4 model calls, so expect **10–40 seconds** per question on your CPU.

---

## How to use this guide with Claude Code

Every step has:

- **Learn:** the idea. Read it first.
- **Predict:** write down what you expect *before* running anything. Being wrong is how you learn.
- **Prompt:** what to type into Claude Code in **plan mode** (`Shift+Tab` until you see `⏸ plan mode on`). Read
  Claude's plan and ask "why?" about anything you don't understand.
- **You check:** what you verify yourself, with the file and function to open.
- **Done when:** a command whose output proves the step works.
- **Commit:** the message.

**Your tutor is still here.** You wrote `.claude/output-styles/tutor.md` in lab 1. Whenever a step says *"Ask your
tutor"*, switch with `/output-style Tutor` first (and back with `/output-style default` before a coding prompt, if
you prefer shorter replies).

**Python is `py` on your machine.** Commands use `py -m ...`; inside Claude Code type `!py -m pytest`. CI on
GitHub runs Linux, where it stays `python`.

---

## Step 0 — Start the branch and check the model can use tools (10 min)

**Learn.** Lab 2 lives in the same `ai-basics` repo as lab 1, on a **branch** called `lab2`. A branch keeps today's
work separate until it's reviewed; at the end you open a **pull request** (PR) and merge it into `main`.

**Optional first (10 min):** lab 1's Parts E and F still need their outputs saved. If you want them done now, open
`notebooks/lab1_how_llms_work.ipynb`, run Parts E–F, save, and commit on `main`
(`Lab 1: save Parts E and F outputs`) before you create the branch. Step 9 writes the README and interview
questions for **both** labs, so nothing else from lab 1 is needed today.

In Git Bash, in your `ai-basics` folder:

```bash
git switch main
git pull
git status
git switch -c lab2
source .venv/Scripts/activate
py -m pip install -e ".[dev,notebook]"
py -m pytest -q
```

**What you should see:** `git status` says `nothing to commit, working tree clean` (commit or stash anything that
isn't). `git switch -c lab2` says `Switched to a new branch 'lab2'`. The tests end with `57 passed, 2 deselected`
(the 2 are the live tests).

Now check that your model supports tools:

```bash
ollama --version
ollama show qwen2.5:3b
```

**What you should see:** a version of `0.12.11` or newer (from lab 1), and in `ollama show`, a **Capabilities**
section that lists `tools` next to `completion`. If `tools` isn't listed, run `ollama pull qwen2.5:3b` again to get
the current version of the model.

Save **this file** as `GUIDE-lab2.md` in the `ai-basics` folder (lab 1's guide stays `GUIDE.md`). Then start
Claude Code:

```bash
claude
```

**Prompt (plan mode):**
> In CLAUDE.md, change the Plan section to say: GUIDE.md is the lab 1 guide and GUIDE-lab2.md is the lab 2 guide
> (tool calling and agents from scratch). Change nothing else.

**Done when:** `!git status` shows `CLAUDE.md` modified and `GUIDE-lab2.md` untracked.

**Commit:** `Add the lab 2 guide` (include GUIDE-lab2.md and CLAUDE.md).

---

## Step 1 — See a tool call in raw JSON, before any Python (15 min)

**Learn.** Re-read section 2c. You'll do the whole loop by hand with `curl`: ask without tools, ask with a tool, then
send the tool's result back. Then you'll print the chat template to see where the tools go.

**Predict.** (1) Without tools, what will the model say about "How many students are in CS 101?" (2) With the
`get_course_info` tool, will `content` have any text in it?

Open a **second** Git Bash window (keep Claude Code open in the first). **Request 1, no tools:**

```bash
curl -s http://localhost:11434/api/chat -d '{"model": "qwen2.5:3b", "stream": false, "options": {"temperature": 0}, "messages": [{"role": "user", "content": "How many students are in CS 101?"}]}' | py -m json.tool
```

**What you should see:** `"message": {"role": "assistant", "content": "..."}`. The model has no data about *your*
CS 101, so it says it doesn't know, or (worse) makes up a number. Either way: no tools, no facts.

**Request 2, with one tool.** This one is long, so it reads the JSON from the lines between `<<'EOF'` and `EOF`
(`-d @-` means "the body comes from standard input"). Copy the whole block:

```bash
curl -s http://localhost:11434/api/chat -d @- <<'EOF' | py -m json.tool
{
  "model": "qwen2.5:3b",
  "stream": false,
  "options": {"temperature": 0},
  "messages": [{"role": "user", "content": "How many students are in CS 101?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_course_info",
      "description": "Look up a course by its code, e.g. 'CS 101'. Returns the course title, number of students and number of sections.",
      "parameters": {
        "type": "object",
        "properties": {"course": {"type": "string", "description": "The course code, e.g. 'CS 101'"}},
        "required": ["course"]
      }
    }
  }]
}
EOF
```

**What you should see:**

```json
"message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {"function": {"name": "get_course_info", "arguments": {"course": "CS 101"}}}
    ]
}
```

(Your Ollama may add an `"index"` or `"id"` field, and the order of keys may differ.) `content` is **empty**: the
model didn't answer, it **asked**. And `arguments` is a JSON **object**, already parsed by Ollama. Also find
`"prompt_eval_count"`. Compare it with request 1: the difference is what the tool description cost in tokens.

**Request 3, send the result back.** Now **you** are the tool: you "ran" `get_course_info` and got 84 students. Add
the model's call and your result to the messages:

```bash
curl -s http://localhost:11434/api/chat -d @- <<'EOF' | py -m json.tool
{
  "model": "qwen2.5:3b",
  "stream": false,
  "options": {"temperature": 0},
  "messages": [
    {"role": "user", "content": "How many students are in CS 101?"},
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_course_info", "arguments": {"course": "CS 101"}}}]},
    {"role": "tool", "tool_name": "get_course_info", "content": "{\"course\": \"CS 101\", \"title\": \"Intro to Programming\", \"students\": 84, \"sections\": 3}"}
  ]
}
EOF
```

**What you should see:** `"content": "CS 101 has 84 students."` (or similar wording), and no `tool_calls`. You just
ran one full turn of an agent loop by hand. Steps 4–5 put exactly this in a `for` loop.

**Now look under the hood.** Print the chat template (the Go template Ollama uses to turn messages into text):

```bash
ollama show qwen2.5:3b --template
```

**What you should see:** a block with `{{- if .Tools }}` that writes `# Tools`, `<tools>`, each tool's JSON, and
instructions to answer inside `<tool_call></tool_call>` tags; and further down, tool results wrapped in
`<tool_response>` tags. That's section 2c in the real file.

**You check:** in request 3's JSON, the `tool` message's `content` is a **string** (with `\"` inside), not an
object. Messages always carry text; your code turns the result into text with `json.dumps`. Ask your tutor:

> I just did a tool call by hand with curl (request 2 and 3 above). Explain what Ollama's chat template and parser
> did with my "tools" field, using the <tool_call> tags I saw in `ollama show qwen2.5:3b --template`. Then quiz me.

**Done when:** you've seen `tool_calls` in request 2 and a text answer with 84 in request 3. (Nothing to commit.)

---

## Step 2 — Part A: tool calling by hand, with no tool API (20 min)

**Learn.** Re-read sections 2a, 2b and 2g. Before using Ollama's `tools` feature in Python, you'll do it the
"handmade" way: describe the tools in plain English in the system message, ask for JSON, and parse it yourself. This
shows that tool calling is just a **text format plus a parser**, and why tool APIs exist.

**Predict.** You'll ask the same question 5 times at temperature 1 and ask for "ONLY JSON". How many of the 5
replies will be valid JSON that your parser accepts?

**Prompt 1 (plan mode):**
> Create ai_basics/tools.py with invented course data COURSES: "CS 101" Intro to Programming, 84 students, 3
> sections; "CS 230" Data Structures, 45, 2; "CS 350" Operating Systems, 36, 1. Two tools:
> - `get_course_info(course)` returns {"course": code, **info}, ignoring case and spaces ("cs101" works), and raises
>   LookupError naming the known courses for an unknown one.
> - `calculate(expression)` does + - * / and parentheses on numbers by walking the `ast` tree. Never eval. Raise
>   ValueError for anything else, including `**` and function calls.
> Also a `TOOLS` dict (name → function) and `TOOL_SCHEMAS` in Ollama's format (type "function", name, description,
> JSON Schema parameters). The calculate description should say to use it for every calculation.
> Then ai_basics/handmade.py: `HANDMADE_SYSTEM`, a system prompt that describes both tools in plain text and says
> to reply with ONLY {"tool": ..., "arguments": {...}} or ONLY {"answer": ...}; and `parse_tool_call(text) ->
> ToolCall | None`, which also accepts a ```json fence. Add `ToolCall(name: str, arguments: dict)` as a frozen
> dataclass in ai_basics/llm.py. Tests in tests/test_tools.py and tests/test_handmade.py, including: "84 / 3" = 28,
> "(84 + 45) / 5" = 25.8, "__import__('os').system('echo hi')" and "2 ** 100" raise ValueError, and
> 'Sure! {"tool": ...}' is not a tool call. Run the tests.

**You check:**
- Open `calculate()` and the helper it calls in `ai_basics/tools.py`. Find the line that accepts a number
  (`ast.Constant`) and the line that maps `ast.Div` to division. Everything not in that list is refused. Ask your
  tutor: *"Draw the ast tree for '(84 + 45) / 5' and walk through how calculate() evaluates it."*
- Open `TOOL_SCHEMAS`: the `name` strings must be exactly the keys of `TOOLS`. If one differs, the model will call a
  name your code can't find. (Ask Claude to add a test for that if it didn't.)
- In `parse_tool_call()`: what happens to `'Sure! {"tool": ...}'`? It returns `None`. We're strict on purpose: a
  parser that "finds JSON somewhere in the text" will also find JSON the user typed.

**Prompt 2 (plan mode):**
> Create notebooks/lab2_tool_calling_and_agents.ipynb. Setup cell like lab 1's (AI_BASICS_PROVIDER, AI_BASICS_MODEL,
> an OllamaClient), plus a helper `model_or_fake(*script)` that returns FakeClient(list(script)) when the provider is
> "fake" and the real client otherwise, so every cell can give CI a scripted reply. Define QUESTION = "How many
> students per section are in CS 101?". Then "## Part A. Tool calling by hand":
> (1) print HANDMADE_SYSTEM; send it with QUESTION; print repr of the raw reply; parse it; if it parsed, print the
> result of TOOLS[call.name](**call.arguments).
> (2) ask the same 5 times at temperature 1.0 with seed=i, print OK or FAIL and the first 70 characters of each
> reply, then "parsed N of 5". Give the fake a mix of good and bad replies.
> Then a markdown cell "**What I learned (Part A):**". Ruff fix and format.

Open the notebook in Jupyter (`py -m jupyter notebook`, in your second Git Bash window) and run the cells.

**What you should see:**
- Cell 1: the model wrote `'{"tool": "get_course_info", "arguments": {"course": "CS 101"}}'` (maybe with a
  ```` ```json ```` fence, which your parser accepts), then `{"course": "CS 101", ..., "students": 84, ...}`. Notice
  the last line runs the Python function **in your notebook**. The model just wrote a string.
- Cell 2: probably 3–5 of 5. Typical failures: text before the JSON, a direct `{"answer": ...}` with a guessed
  number, or a missing brace. At temperature 1 every extra word is a chance to break the format (lab 1, section 2c).

**You check:** in your "What I learned" cell, write one line: what went wrong in each FAIL? Then ask your tutor:

> Look at my Part A outputs in @notebooks/lab2_tool_calling_and_agents.ipynb. Why do small models break a
> "reply with only JSON" rule, and which of the three pieces in a real tool API (training, chat template, parser)
> fixes each failure I got?

**Done when:** `!py -m pytest -q` shows `0 failed`, and the notebook is saved with Part A outputs.

**Commit:** `Part A: tools and tool calling by hand`

---

## Step 3 — Part B: tool calling through the API (15 min)

**Learn.** Now let Ollama do the format and parsing (section 2c). Your client needs three changes: send `tools`,
read `tool_calls` back, and turn the model's reply into a history message that **keeps** its tool calls (section
2d: the model must see its own call before the result).

**Predict.** Your two tool descriptions will be added to every request. How many extra input tokens: about 20,
about 200, or about 2,000?

**Prompt 1 (plan mode):**
> Add tool support to `chat()` in ai_basics/llm.py, in the Protocol, OllamaClient and FakeClient:
> - a keyword `tools: Sequence[dict] | None = None`; send "tools" only when given.
> - `LLMResponse.tool_calls: list[ToolCall]` (default empty), parsed from message.tool_calls. Arguments come as a
>   JSON object, but also accept a JSON string (OpenAI sends a string).
> - `Message`: add role "tool" and optional keys `tool_calls` and `tool_name` (NotRequired).
> - `assistant_message(response) -> Message`: the reply as a history message, keeping tool calls in Ollama's format.
> - FakeClient: a scripted reply can also be a list[ToolCall] (text ""); record `tools` in FakeCall; record a copy of
>   the messages, not the caller's list.
> Tests with MockTransport and FakeClient. All existing tests must still pass. Then add a notebook section
> "## Part B. The tool API": ask QUESTION as a single user message without tools and with TOOL_SCHEMAS; print repr
> of each text, the tool_calls, both input token counts, and the difference ("the tool descriptions cost N
> tokens"), plus a "What I learned (Part B)" cell. Ruff fix and format.

**You check:**
- In `OllamaClient.chat()`: `"tools"` is added to the body **only** when tools are given. Lab 1's tests still pass,
  which proves the old requests didn't change.
- Why does FakeClient record a **copy** of the messages? Ask Claude, or think it through: the agent keeps appending
  to one list. If FakeClient kept a reference to that same list, every recorded call would show the *final* list,
  and you couldn't test what the model saw at step 1. (Python lists are passed by reference, like objects in Java.)
- `assistant_message()` in `ai_basics/llm.py`: compare its output with the `assistant` message you typed by hand in
  Step 1's request 3. It should be the same shape.

Restart the Jupyter kernel (**Kernel → Restart**, because `llm.py` changed), then run all cells.

**What you should see:** without tools, the model writes words (it doesn't know CS 101). With tools, the text is
`''` and there's `[ToolCall(name='get_course_info', arguments={'course': 'CS 101'})]`. The tool descriptions
probably cost **200–300 tokens**, on every request, before the user types anything.

**Done when:** `!py -m pytest -q` shows `0 failed`, and the notebook is saved with Part B outputs.

**Commit:** `Part B: tool calling through the chat API`

> Good place to stop for today. Next time, re-read section 2d, then start at Step 4.

---

## Step 4 — You type `run_tool`: from a name to a function call (10 min)

**Learn.** The model gives you a `ToolCall` with a **name** (a string) and **arguments** (a dict). `run_tool` is the
**dispatcher**: it finds the Python function by name in `TOOLS` and calls it. Two Python ideas do all the work:

- `tools["calculate"]` is the function object itself (functions are values in Python, like in JavaScript).
- `f(**{"expression": "84 / 3"})` is the same as `f(expression="84 / 3")`: `**` unpacks a dict into keyword
  arguments.

The important design choice: **errors become text, not exceptions.** An unknown tool, a wrong argument name or a
failing lookup are all things the model can fix if it reads about them. A crash would end the whole agent.

**Predict.** What will `run_tool` return for each of these?
1. `ToolCall("calculate", {"expression": "84 / 3"})`
2. `ToolCall("send_email", {"to": "class"})`
3. `ToolCall("calculate", {"expr": "84 / 3"})` (wrong argument name)

**Prompt (plan mode):**
> Create ai_basics/agent.py with:
> - `AGENT_SYSTEM`: "You are a course assistant. Use the tools to look up facts and to do arithmetic; never guess a
>   number. When you have the answer, reply in one short sentence."
> - a dataclass `AgentResult(answer: str, stopped: Literal["answer", "max_steps"], steps: int, input_tokens: int,
>   messages: list[Message])`
> - `show_message(message)`: prints one line: "[tool → name] content" for tool messages, "[assistant wants]
>   name({json args})" for each tool call, otherwise "[role] content", each indented two spaces.
> - `run_tool(call: ToolCall, tools: dict[str, Callable]) -> str` and `run_agent(client, question, *, tools=TOOLS,
>   schemas=TOOL_SCHEMAS, max_steps=5, verbose=False) -> AgentResult`, both with the body `raise
>   NotImplementedError` (I'll write them).
> Write tests first in tests/test_agent.py with my guide's paper example (CS 101: 84 students, 3 sections):
> - run_tool: the lookup returns JSON text with 84 and 3; calculate "84 / 3" returns "28.0"; an unknown tool returns
>   text starting "error: unknown tool 'send_email'" that lists both tool names; a wrong argument name returns
>   "error: TypeError..."; CS 999 returns "error: LookupError..." mentioning CS 101. Name these tests test_run_tool_*.
> - run_agent with FakeClient scripted [[lookup CS 101], [calculate "84 / 3"], "CS 101 has 28 students per
>   section."]: answer, stopped "answer", 3 steps, input_tokens 300, roles [system, user, assistant, tool, assistant,
>   tool, assistant], and message 5 == {"role": "tool", "tool_name": "calculate", "content": "28.0"}.
> - every model call gets the tools, and the history lengths per call are [2, 4, 6].
> - the model sees its own tool call right before the result; a text reply stops after 1 step; max_steps=2 with a
>   model that only calls tools stops with "max_steps" after 2 calls and the answer mentions "(2)"; a CS 999 error
>   reaches the model; two tool calls in one reply give two tool messages (84 and 45); verbose prints "step 1",
>   "[assistant wants] get_course_info(...)", "[tool → calculate] 28.0".
> - the loop also works with OllamaClient on a MockTransport that replays three Ollama replies.
> Run the tests and show me them failing.

**You check:** read the test names in `tests/test_agent.py`. Each one is a sentence about how an agent should behave.
Find the one with `[2, 4, 6]` and make sure you can explain it from section 2d.

**Now you type** the body of `run_tool` in `ai_basics/agent.py`, in your editor:

```python
def run_tool(call: ToolCall, tools: dict[str, Callable[..., Any]]) -> str:
    """Run one tool call and return the result as text. Errors become text too, for the model."""
    if call.name not in tools:
        return f"error: unknown tool {call.name!r}. Available tools: {', '.join(tools)}"
    try:
        result = tools[call.name](**call.arguments)
    except Exception as exc:  # bad arguments, unknown course, division by zero...
        return f"error: {type(exc).__name__}: {exc}"
    return json.dumps(result)
```

Read it line by line:
- `call.name not in tools`: the model can invent a tool name. Telling it which tools exist lets it recover.
- `tools[call.name](**call.arguments)`: look up, then call with the model's arguments as keyword arguments. A wrong
  argument name raises `TypeError` here, which the next line catches.
- `except Exception`: usually too broad in normal code. Here it's on purpose: **any** failure becomes a message the
  model can read. (It doesn't catch `KeyboardInterrupt`, so `Ctrl+C` still stops the program.)
- `type(exc).__name__`: the class name, like `LookupError`. It helps the model understand what kind of mistake it
  made.
- `json.dumps(result)`: messages carry text, so the dict becomes a JSON string (like in Step 1, request 3), and the
  float `28.0` becomes `"28.0"`.

(Check the imports at the top of `agent.py`: you need `json`, `Callable` and `Any` here, and `assistant_message`,
`Message`, `TOOLS` and `TOOL_SCHEMAS` for Step 5. Claude probably added them.)

```bash
py -m pytest -q tests/test_agent.py -k run_tool
```

**Done when:** the `run_tool` tests pass (the `run_agent` ones still fail: that's Step 5).

**Commit:** `run_tool: dispatch by name, errors become text`

---

## Step 5 — You type `run_agent`: the loop (20 min)

**Learn.** Re-read sections 2d, 2e and 2f, and keep the paper trace (rows 0–6) next to you. You're about to type it
as code. Every line of `run_agent` matches a line of the pseudocode in 2d.

**Predict.** For the paper example: how many times does the `for` loop run? How many times does the inner
`for call in response.tool_calls` loop run in total? What is `input_tokens` with FakeClient (100 per call)?

**Now you type** the body of `run_agent` in `ai_basics/agent.py`:

```python
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
    messages: list[Message] = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": question},
    ]
    input_tokens = 0
    for step in range(1, max_steps + 1):
        response = client.chat(messages, tools=schemas)
        input_tokens += response.input_tokens
        messages.append(assistant_message(response))
        if verbose:
            print(f"step {step} ({response.input_tokens} input tokens)")
            show_message(messages[-1])

        if not response.tool_calls:  # stop 1: the model answered in words
            return AgentResult(response.text, "answer", step, input_tokens, messages)

        for call in response.tool_calls:
            messages.append(
                {"role": "tool", "tool_name": call.name, "content": run_tool(call, tools)}
            )
            if verbose:
                show_message(messages[-1])

    # stop 2: the safety limit
    return AgentResult(
        f"I hit the step limit ({max_steps}) before finding an answer.",
        "max_steps",
        max_steps,
        input_tokens,
        messages,
    )
```

Read it line by line, against the paper trace:
- `messages = [system, user]`: rows 0 and 1. This list is the agent's **only memory**.
- `for step in range(1, max_steps + 1)`: at most 5 model calls. This is stop 2. Without it, a confused model could
  loop forever (and on a hosted API, spend money forever).
- `client.chat(messages, tools=schemas)`: the **whole** history and the **same** tool list, every time (section 2f).
- `messages.append(assistant_message(response))`: rows 2, 4 and 6. It goes in **before** the tool results.
- `if not response.tool_calls: return ...`: stop 1. A reply with no tool calls is the final answer (row 6).
- The inner `for` loop: a reply can contain **several** calls (parallel tool calls); each gets its own `tool`
  message (rows 3 and 5), with `tool_name` so the model knows which result is which.
- `"tool_name": call.name`: Ollama's way to match a result to its call. OpenAI and Anthropic use an id instead
  (overview section 7).

That's the whole agent: about 40 lines, and most of them are printing and bookkeeping.

```bash
py -m pytest -q tests/test_agent.py
py -m pytest -q
```

**What you should see:** all of `tests/test_agent.py` passes, then the whole suite: `0 failed`.

**Break it on purpose** (Level-up tip 2 from Week 1). Make each change, run `py -m pytest -q tests/test_agent.py`,
note which test fails, then undo it (`git checkout ai_basics/agent.py` puts back your committed version, so
**commit first**):

1. Delete the line `messages.append(assistant_message(response))`. (Predict: which test fails, and why would a real
   model get confused?)
2. Change `if not response.tool_calls:` to `if response.text:`. (Predict: will any test fail? Run it and see.
   Probably **none** does, because your FakeClient never writes text **and** a tool call in one reply. A real model
   can: "Let me look that up" plus a call. This version would then stop early and return "Let me look that up" as
   the answer. Tests only catch what they script; write this gap down for the agent evals on Friday.)
3. Change `range(1, max_steps + 1)` to `range(1, max_steps)`.

**You check:** ask your tutor:

> Here is my agent loop: @ai_basics/agent.py. Walk through it with my paper example (CS 101: lookup, then 84 / 3,
> then the answer), showing the messages list after each step. Then quiz me on stopping conditions.

**Commit:** `run_agent: the agent loop with two stopping conditions` (commit before the break-it experiments, then
check `git status` is clean after undoing them).

---

## Step 6 — Part C: run the agent on the real model and read every message (10 min)

**Learn.** The tests proved the loop with a scripted model. Now the real model makes its own choices. Watch three
things: **which tools it picks**, **the arguments it writes**, and **how the input tokens grow each step** (section
2f).

**Predict.** For QUESTION (CS 101 per section): how many steps will the real model take? Will it use `calculate` for
84 / 3, or just write 28? What will step 3's input token count be compared with step 1's?

**Prompt (plan mode):**
> Add "## Part C. The agent loop" to the notebook: run run_agent(llm, QUESTION, verbose=True) with a fake script
> for CI (lookup CS 101, calculate "84 / 3", the answer), then print answer, stopped, model calls and total input
> tokens. A second cell prints every message in result.messages with its index. Then a "What I learned (Part C)"
> cell. Add a live test in tests/test_live.py: the agent's answer to QUESTION contains "28", it stopped with
> "answer", and at least one tool message is in the history. Ruff fix and format.

Restart the kernel and run Part C. Then the live tests, in Git Bash:

```bash
py -m pytest -m live -v
```

**What you should see** (your wording and numbers will differ):

```text
step 1 (330 input tokens)
  [assistant wants] get_course_info({"course": "CS 101"})
  [tool → get_course_info] {"course": "CS 101", "title": "Intro to Programming", "students": 84, "sections": 3}
step 2 (400 input tokens)
  [assistant wants] calculate({"expression": "84 / 3"})
  [tool → calculate] 28.0
step 3 (440 input tokens)
  [assistant] CS 101 has 28 students per section.
```

Probably 3 steps, and each step's input count is **bigger** than the one before. A small model might skip
`calculate` and write 28 itself (2 steps), or write `"84/3"` without spaces (fine: `ast` doesn't care). The live
tests: `3 passed`.

**You check:**
- Compare your real token numbers with the paper example in section 2f. Add them up: is the total about 3× step 1?
- In the second cell, find the `assistant` message with `tool_calls`. Its `content` is `''`. Now you've seen, in
  your own data, that "the model asked for a tool" and "the model answered" are different kinds of reply.
- Write in "What I learned": the step count, the total input tokens, and one choice the model made that surprised
  you.

**Done when:** the live tests pass, and the notebook is saved with Part C outputs.

**Commit:** `Part C: run the agent on the real model`

---

## Step 7 — Part D: stopping conditions, broken on purpose (15 min)

**Learn.** Re-read section 2e. A working agent is the easy part; interviewers ask what happens when things go wrong.
You'll run four experiments. The last one uses a question that needs three lookups and a calculation. On paper:
**(84 + 45 + 36) × 37 = 165 × 37 = 6,105**.

**Predict** (write all four down before you run anything):
- D1: `max_steps=1` on QUESTION. What does `stopped` say, and what's the answer?
- D2: "How many students per section are in CS 999?" What will the model do after the tool error?
- D3: "What does CPU stand for?" How many steps?
- D4: "Every student in CS 101, CS 230 and CS 350 buys a $37 textbook. How much is spent in total?" With the
  calculator, and then **without** it. Will the model get 6,105 both times? Will it look up all three courses in
  **one** step (parallel calls) or three?

**Prompt (plan mode):**
> Add "## Part D. Stopping conditions" to the notebook, one cell per experiment, each with verbose=True and a fake
> script for CI, printing the stop reason and the answer:
> D1 QUESTION with max_steps=1. D2 "How many students per section are in CS 999?". D3 "What does CPU stand for?".
> D4 the question "Every student in CS 101, CS 230 and CS 350 buys a $37 textbook. How much is spent in total?",
> run twice with max_steps=8: with all tools, and with only get_course_info (tools and schemas both limited).
> Print both answers at the end. For D4 the fake should do three lookups in one reply. Then a
> "What I learned (Part D)" cell. Ruff fix and format.

Restart the kernel and run Part D.

**What you should see** (the real model may differ):
- **D1:** one step with a tool call, then `max_steps | I hit the step limit (1) before finding an answer.` The model
  never got to read the lookup result. The step limit protected you, but the user got nothing useful.
- **D2:** `[tool → get_course_info] error: LookupError: no course 'CS 999'; known courses: CS 101, CS 230, CS 350`,
  then the model tells the user there's no such course (and may list the real ones). The error message **was** the
  help.
- **D3:** probably one step, no tools, `stopped: answer`. (If your model calls a tool anyway, that's a known
  weakness of small models with tools: write it down.)
- **D4:** with the calculator: lookups (maybe three calls in one step), then something like
  `calculate({"expression": "(84 + 45 + 36) * 37"})` → `6105`, and the right total. Without it: the model has to do
  165 × 37 itself, token by token. A 3B model often gets it wrong (lab 1: it never "sees" a number as a number).

**You check:** for each experiment, compare with your prediction and write one line in "What I learned". Then ask
your tutor:

> Using my Part D outputs in @notebooks/lab2_tool_calling_and_agents.ipynb: which stopping conditions did I see,
> what would a production agent add (time limit, token budget, human approval), and why does D4 show that tools
> make a model more reliable, not just more capable? Then quiz me.

**Done when:** the notebook is saved with Part D outputs, and `!py -m pytest -q` shows `0 failed`.

**Commit:** `Part D: stopping conditions and a missing tool`

---

## Step 8 — Claude Code technique: watch Claude Code's own agent loop (10 min)

**Learn.** Re-read overview section 6. You've built an agent; Claude Code is one too. Today you look at its loop two
ways: the **transcript viewer** inside a session, and a **headless** run that prints every message as JSON.

**Predict.** You'll ask Claude Code: "Count the .py files in ai_basics/." How many model calls and tool calls will
it make? Which tool will it use? (Count the files yourself first: you should get 7.)

**Part 1, the transcript viewer.** In your Claude Code session (normal mode, not plan mode), type:

> Count the .py files in ai_basics/ and tell me their names. Don't change anything.

When it's done, press **`Ctrl+O`**. **What you should see:** the detailed view: each tool call (probably `Glob` or
`Bash` with `ls`), its full result, then Claude's answer, with a timestamp and the model name on each assistant
message. Press `Esc` to go back. Compare it with your `verbose=True` output from Part C: same shape (assistant
asks → tool result → assistant answers).

**Part 2, every message as JSON.** In your second Git Bash window, in the `ai-basics` folder:

```bash
claude -p "Use the Glob tool to count the .py files in ai_basics/, then reply with just the number." --output-format stream-json --verbose > cc_stream.jsonl
py -c "
import json
for line in open('cc_stream.jsonl', encoding='utf-8'):
    event = json.loads(line)
    content = event.get('message', {}).get('content', [])
    blocks = [block.get('type') for block in content] if isinstance(content, list) else []
    print(event['type'], blocks)
"
```

`-p` ("print" mode) runs one prompt with no interactive session and exits; it's also how CI will run Claude Code
on Dec 3. Each line of the file is one JSON event; the Python snippet prints each event's type and the
kinds of content blocks inside it.

**What you should see:** something like

```text
system []
assistant ['tool_use']
user ['tool_result']
assistant ['text']
result []
```

(You may also see `thinking` blocks, or extra events, depending on your version.) Read it against your agent:

- `assistant ['tool_use']` is your `[assistant wants] ...` line: the model asked for `Glob`.
- `user ['tool_result']`: Anthropic's format puts tool results in a **user** message (overview section 7), where
  yours uses role `"tool"`. Same idea.
- `assistant ['text']`: no tool use, so the loop stopped. That's your stop 1.
- `result`: the summary at the end (it includes the cost and the number of turns).

Look at one `tool_use` block in full:

```bash
grep '"tool_use"' cc_stream.jsonl | head -1 | py -m json.tool
```

Find the `"id"` (it starts with `toolu_`), `"name": "Glob"` and `"input"`, and then find the same id as
`"tool_use_id"` in the `tool_result` line. That's the id matching from overview section 7, which your Ollama agent
does by `tool_name`.

Then delete the file (it's a session log, not code):

```bash
rm cc_stream.jsonl
```

**You check:** fill in this table in your notebook's last markdown cell (or on paper):

| | my agent | Claude Code |
|---|---|---|
| model calls for one simple question | | |
| how tool results are sent back | role "tool" + tool_name | |
| what stops the loop | | |
| who approves a tool before it runs | nobody (today) | |

**Done when:** you've seen `tool_use` → `tool_result` → `text` in the JSON, and filled in the table.

**Commit:** `Compare my agent loop with Claude Code's` (if you put the table in the notebook).

---

## Step 9 — CI, README, interview questions, pull request (15 min)

**Prompt 1 (plan mode):**
> In .github/workflows/ci.yml add a last step "Notebooks run top to bottom (fake model)" with env
> AI_BASICS_PROVIDER=fake and MPLBACKEND=Agg that runs `python -m jupyter nbconvert --to notebook --execute
> --output-dir /tmp` on notebooks/lab1_how_llms_work.ipynb and notebooks/lab2_tool_calling_and_agents.ipynb.

**Prompt 2 (plan mode):**
> Rewrite README.md for both labs. Remove the "How I used Claude Code" section. "Run it": install, tests, live
> tests (`py -m pytest -m live`, needs Ollama and qwen2.5:3b), and how to open the notebooks (and run them with
> AI_BASICS_PROVIDER=fake). A section "Lab 1: how an LLM works" with a table of Parts A–F (question, experiment, my
> result) from the lab 1 notebook outputs, and a section "Lab 2: tool calling and agents" with a table of Parts A–D
> from the lab 2 notebook outputs, using my real numbers. Fill "Design decisions" for both labs; for lab 2: tool
> calling by hand first, then the API, so the mechanism is visible; a ~40-line agent loop with no framework; tool
> errors returned to the model as text; two stopping conditions (answer, max_steps); calculate() walks the ast and
> never evals model output; FakeClient scripts tool calls so CI runs the notebooks. "Known limitations" for lab 2:
> one small model on a CPU; no human approval, time limit or token budget; tool results are matched by name, not
> id; the whole history is resent every step (no caching or trimming); invented course data.

**Prompt 3 (plan mode):**
> Replace the TODOs in @INTERVIEW.md with 10 questions and model answers (5–8 sentences each, plain language), under
> two headings, "Lab 1" and "Lab 2", based on this repo and my notebook outputs. Lab 1: the five questions in
> GUIDE.md Step 9. Lab 2:
> 6. What is tool calling? Who actually runs the tool, and how does the model know which tools exist?
> 7. Walk me through an agent loop. How does it stop, and what happens when a tool fails?
> 8. Why does an agent use more tokens than a single call, and how would you reduce it?
> 9. The model chooses the tool arguments. What can go wrong, and how do you make tools safe?
> 10. LLM system design: design an agent that helps instructors answer routine student questions about a course
>     (deadlines, syllabus, office hours) by looking things up in the course system and drafting a reply. Cover
>     the tools, permissions (read vs. write), a human approving anything sent, stopping conditions, how you test
>     it before launch (agent evals), and what you monitor after.

**You check:** every number in the README tables is one you actually saw. Rewrite the answer to Q8 in your own words,
using your Part C token numbers.

**Done when:**

```bash
py -m ruff check . && py -m ruff format --check . && py -m pytest -q
```

all pass. **Commit:** `Document both labs in README and interview Q&A`

Push the branch and open a pull request:

```text
!git push -u origin lab2
!gh pr create --base main --title "Lab 2: tool calling and agents from scratch" --body "Tool calling by hand and through the API, a 40-line agent loop with two tools and two stopping conditions, experiments on the real model, and README/INTERVIEW for both labs."
!gh pr checks --watch
```

**What you should see:** a PR link, then the checks running and finishing with `pass`. If the notebook step fails,
open the PR on GitHub, click the failed check, read the error at the bottom of that step, and ask Claude to fix it.
Leave the PR open: when you tell the roadmap chat "done", it reviews the PR before you merge it.

---

## Check your understanding (for the reading day)

Answer in your own words first, then open the answer.

1. A student says "the model ran the calculator." What's wrong with that sentence?
   <details><summary>Answer</summary>The model only wrote text: a tool name and arguments in an agreed JSON format.
   Your code (run_tool) looked up the Python function, ran it, and sent the result back as a message. The model
   never executes anything, which is why your code is where safety checks belong.</details>
2. Trace the agent on paper for "How many students per section are in CS 230?" (45 students, 2 sections). How many
   model calls, tool runs and messages, and what's the answer?
   <details><summary>Answer</summary>Same shape as CS 101: lookup → 45 and 2; calculate "45 / 2" → 22.5; answer.
   3 model calls, 2 tool runs, 7 messages (system, user, assistant, tool, assistant, tool, assistant). Answer: 22.5
   students per section (a real reply might say "about 22 or 23").</details>
3. Why does the assistant message with the tool call go into the history **before** the tool result? What breaks
   if you leave it out?
   <details><summary>Answer</summary>The model has no memory between calls; it only knows what's in the messages.
   Without its own tool-call message, it sees a result for a call it never made, may not know what the result is
   for, and often calls the same tool again. In Step 5's break-it experiment, the roles and history-length tests failed.
   </details>
4. System + tools + question = 400 tokens; each tool call adds 20 and each result 40. Two tool calls, then an
   answer. What are the input tokens per step, and in total?
   <details><summary>Answer</summary>Step 1: 400. Step 2: 400 + 20 + 40 = 460. Step 3: 460 + 20 + 40 = 520. Total:
   1,380 input tokens, about 3.5× one call. Each step re-sends everything before it.</details>
5. Why does `run_tool` return `"error: ..."` text instead of raising the exception?
   <details><summary>Answer</summary>Most tool errors are mistakes the model can fix: a wrong course code, a
   wrong argument name, a tool that doesn't exist. As text in the history, the error tells the model what went
   wrong (and, for a bad course, which ones exist), so it can retry or explain it to the user. An exception would
   crash the whole run and throw away everything done so far.</details>
6. What does `max_steps` protect against? Name two more limits a production agent should have.
   <details><summary>Answer</summary>A model that never stops calling tools (stuck in a loop, or a task too big),
   which would run forever and, on a hosted API, keep spending money. Production agents also add a time limit, a
   token or cost budget, and a human approval step before any action with side effects (sending, deleting,
   paying).</details>
7. Why does `calculate()` walk the `ast` tree instead of calling `eval()`? Who could attack it, if the model writes
   the expression?
   <details><summary>Answer</summary>eval runs any Python, including `__import__('os').system(...)`. The model
   writes the expression, and the model's input can include text from users or documents, so someone could write a
   note that says "calculate __import__('os')..." and the model might pass it on (prompt injection, Nov 27).
   Walking the ast allows only numbers and + - * /, and refuses everything else without running it.</details>
8. Your triage CLI (Nov 9) always does classify → validate → log. Is it an agent? When would you make it one?
   <details><summary>Answer</summary>No, it's a workflow: your code fixes the steps and the model fills in one of
   them. You'd make it an agent only if the next step depended on what it found, for example "look up the student's
   course, check the deadline, then decide whether it needs a human". Workflows are cheaper, faster and easier to
   test, so use them when you know the steps.</details>

---

## When you're done

Tell the roadmap chat **"done"**. It will review the PR, run the tests and tick this row.

## Level up

**Read the loop, not just the answer.** Every agent, yours or Claude Code, is a list of messages: the model asks
for a tool, a result comes back, repeat. When Claude Code does something surprising, press `Ctrl+O` and read the
tool calls and results it based its choice on, the same way you read your `verbose=True` output, before you
re-prompt. For a record you can search or diff, run it headless with `claude -p ... --output-format stream-json
--verbose` and read the `tool_use` and `tool_result` events. And when you design your own agent, decide the stops
first (a step limit, errors as text, a human before anything with side effects), then add tools.
