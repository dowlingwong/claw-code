# Ollama Integration

This document describes the current local Ollama integration in the Python `claw-code` workspace, the features the harness uses today, and the commands to exercise them.

It is intended as a reference for future API cleanup, provider abstraction work, or deeper tool integrations.

## Scope

The current Ollama integration lives in the Python workspace:

- [`src/llm_backend.py`](./src/llm_backend.py): low-level `/api/chat` client
- [`src/task.py`](./src/task.py): one-shot local chat wrapper
- [`src/runtime_tools.py`](./src/runtime_tools.py): executable workspace tools
- [`src/agent_loop.py`](./src/agent_loop.py): multi-turn agent loop with tool use
- [`src/main.py`](./src/main.py): CLI entrypoints

## Default Configuration

- Host: `http://localhost:11434`
- Default model: `qwen2.5-coder:7b`
- Endpoint: `POST /api/chat`
- Transport: one non-streaming request per turn

## Ollama Features Currently Used

### 1. Chat messages

The harness sends standard Ollama chat messages:

```json
{
  "model": "qwen2.5-coder:7b",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "stream": false
}
```

Used by:

- `python3 -m src.main chat ...`
- `python3 -m src.main agent ...`

### 2. Non-streaming responses

The harness currently uses `stream: false`.

That means:

- each turn waits for a full Ollama response
- the Python agent loop is simpler
- there is no token-by-token terminal rendering yet

This is fine for early harness integration, but future work could add streaming for better UX and partial tool-call parsing.

### 3. JSON response mode

The agent path asks Ollama for JSON output:

```json
{
  "format": "json"
}
```

This is only used in agent mode. It makes the model return structured responses like:

```json
{"type":"tool_call","name":"read_file","arguments":{"path":"src/main.py"}}
```

or:

```json
{"type":"final","content":"..."}
```

### 4. Ollama options

The agent path currently sets:

```json
{
  "options": {
    "temperature": 0
  }
}
```

Reason:

- lower randomness improves protocol adherence
- weaker local models are less likely to drift away from the JSON tool format

### 5. Multi-turn conversation state

The one-shot `chat` command sends a single system+user turn.

The `agent` command maintains a growing message list across turns:

- system instructions
- original user prompt
- assistant JSON tool call
- synthetic user `TOOL_RESULT ...` message
- final assistant answer

This is the core mechanism that lets Ollama use local harness tools.

## Tool Protocol

Agent mode expects exactly one JSON object from the model on each turn.

Valid shapes:

```json
{"type":"tool_call","name":"read_file","arguments":{"path":"src/main.py"}}
```

```json
{"type":"final","content":"Summary goes here."}
```

If the model returns invalid JSON or tries to answer directly without using a required tool, the harness will:

1. reject the response
2. ask the model to follow the protocol again
3. for obvious prompt shapes, synthesize a fallback first tool call

## Executable Tools Available Today

The current agent exposes these real tools:

### `read_file`

Arguments:

- `path`
- optional `start_line`
- optional `end_line`

Behavior:

- reads a file inside the workspace root
- returns numbered lines
- rejects paths outside the workspace

### `search_files`

Arguments:

- `pattern`
- optional `path`
- optional `max_results`

Behavior:

- uses `rg` when available
- falls back to a Python file scan otherwise
- returns `path:line:content` matches

### `edit_file`

Arguments:

- `path`
- `new_text`
- optional `old_text`

Behavior:

- creates a new file when the target does not exist and `old_text` is omitted
- replaces exactly one matching snippet in an existing file
- rejects ambiguous multi-match replacements

### `run_shell_command`

Arguments:

- `command`
- optional `timeout_seconds`

Behavior:

- runs inside the workspace root
- captures stdout, stderr, and exit code
- marks non-zero exit as failure

## CLI Commands

### One-shot local chat

Use this when you only want a plain local model response and do not need tool use.

```bash
python3 -m src.main chat "write a hello world in python"
```

Override model or host:

```bash
python3 -m src.main chat --model qwen2.5-coder:7b --host http://localhost:11434 "summarize this repo"
```

Pipe stdin:

```bash
printf 'summarize src/main.py' | python3 -m src.main chat
```

### Agent mode with tool use

Use this when the task requires reading files, searching code, editing files, or running commands.

```bash
python3 -m src.main agent "read src/main.py lines 1 to 40 and summarize the CLI"
```

Search the workspace:

```bash
python3 -m src.main agent "search the src tree for 'agent_parser' and tell me which file contains it"
```

Edit a file:

```bash
python3 -m src.main agent "create tmp_test.txt containing hello harness"
```

Run shell commands:

```bash
python3 -m src.main agent "run 'pwd' and 'ls src | head' and summarize the results"
```

### Agent trace mode

Use `--trace` to verify that the model actually called a harness tool.

```bash
python3 -m src.main agent --trace "read src/main.py lines 1 to 40 and quote the subcommands you find"
```

Expected stderr output looks like:

```text
[trace] assistant_raw[1]={"type":"tool_call","name":"read_file","arguments":{"path":"src/main.py","start_line":1,"end_line":40}}
[trace] tool_call[1]=read_file {"path": "src/main.py", "start_line": 1, "end_line": 40}
[trace] tool_result[1]=read_file success=True error=None
[trace] assistant_raw[2]={"type":"final","content":"..."}
```

### Deny tools

You can block specific tools or prefixes in agent mode:

```bash
python3 -m src.main agent --deny-tool run_shell_command "inspect this repo"
```

```bash
python3 -m src.main agent --deny-prefix read "inspect src/main.py"
```

## Prompt Shapes That Work Best

Local models follow the protocol more reliably when prompts are concrete.

Good:

- `read src/main.py lines 1 to 40 and list the subcommands`
- `search the src tree for 'agent_parser'`
- `run 'pwd' and 'ls src | head' and summarize the results`
- `create tmp_test.txt containing hello harness`

Weaker:

- `inspect the codebase`
- `look around and tell me what you think`

Reason:

- concrete prompts make it easier for the harness to infer the correct first tool call when the model hesitates

## Current Limitations

- no streaming responses yet
- no native Ollama function-calling schema; this is prompt+JSON protocol driven
- no patch-based editing tool yet, only exact text replacement or file creation
- shell execution is real but still minimal; there is no deep safety policy beyond workspace scoping and CLI deny lists
- the agent does not yet plan complex multi-step edits as well as a stronger hosted model

## Suggested Next API Improvements

If you extend this integration later, these are the next sensible upgrades:

1. Add streaming support to `/api/chat`
2. Separate model protocol logic from tool orchestration
3. Introduce a stronger typed schema for tool calls and tool results
4. Add patch-style file editing instead of exact-text replacement only
5. Add richer shell permission policies
6. Store traces and tool transcripts in sessions
7. Move Ollama into a provider abstraction so local and hosted models share one agent runtime

## Quick Verification Checklist

Before building on this integration, verify:

1. `ollama serve` is running
2. the target model is available locally
3. `python3 -m src.main chat "hello"` works
4. `python3 -m src.main agent --trace "read src/main.py lines 1 to 20"` shows a real `tool_call`

## Example End-to-End Session

Command:

```bash
python3 -m src.main agent --trace "search the src tree for 'agent_parser' and tell me which file contains it"
```

Observed behavior:

1. Ollama returns `{"type":"tool_call","name":"search_files",...}`
2. the harness executes `search_files`
3. the harness feeds back `TOOL_RESULT {...}`
4. Ollama returns `{"type":"final","content":"..."}`

That is the current definition of "real harness tool use" for the local Ollama path.
