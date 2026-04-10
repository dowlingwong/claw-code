# Stage 1 Demo — Implementation Checklist

Agent instructions: work through each task in order. Each task is self-contained. The acceptance criterion tells you exactly when to stop. Commit after each task.

---

## Gap Summary

The plan requires: `GET /memory`, a concurrent-run guard, `rationale` on keep/discard, a structured memory schema, a manager loop prompt, and a demo launcher script.

What exists today:
- ✅ Control plane REST API (`src/api_server.py`) with `/setup /isolate /baseline /run /keep /discard /loop /status /health /log`
- ✅ Experiment state machine (`src/autoresearch_worker.py`)
- ✅ Append-only memory log (`experiment_memory.jsonl`)
- ✅ Worker (local LLM via `src/agent_loop.py`)
- ✅ Rust claw ↔ Python API bridge tested in `notebooks/autoresearch_adapter_demo.ipynb`
- ❌ `GET /memory` endpoint (plan requires manager to read it)
- ❌ Concurrent-run guard on `POST /run` (plan requires 409 if run already pending)
- ❌ `rationale` field accepted and persisted on `POST /keep` and `POST /discard`
- ❌ Structured memory schema (current entries lack `experiment_id`, `code_change_summary`, `decision_rationale`, `failure_tag`)
- ❌ Manager system prompt (plan requires manager to read `/status` → `/memory` → `POST /run` → decide keep/discard)
- ❌ Demo launcher script (one command to start server + run one full loop)

---

## Task 1 — Add `GET /memory` endpoint

**File:** `src/api_server.py`

**What to add:** Handle `GET /memory` in the request router (alongside the existing `GET /status` and `GET /log` handlers). Read `experiment_memory.jsonl` from the node root and return its lines as a JSON array of objects.

Implementation sketch:
```python
# In the GET dispatch block, add:
elif path == '/memory':
    mem_path = self._node_root / 'experiment_memory.jsonl'
    if not mem_path.exists():
        self._send_json({'memory': []})
        return
    entries = []
    for line in mem_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    self._send_json({'memory': entries})
```

**Acceptance criterion:** `curl http://localhost:PORT/memory` returns `{"memory": [...]}` (empty list if no experiments have run, non-empty list after a run).

---

## Task 2 — Add concurrent-run guard on `POST /run`

**File:** `src/api_server.py` and/or `src/autoresearch_worker.py`

**What to add:** Before starting a new experiment in the `POST /run` handler, check whether `.autoresearch_state.json` already contains `state: "pending"`. If it does, respond with HTTP 409 and a JSON body `{"error": "experiment already pending", "state": "pending"}` without starting a new run.

Implementation sketch (in `api_server.py` POST /run handler):
```python
state = load_autoresearch_state(node_root)
if state.get('state') == 'pending':
    self._send_json({'error': 'experiment already pending', 'state': 'pending'}, status=409)
    return
```

**Acceptance criterion:** Start a run, then immediately send a second `POST /run` before the first completes — the second call returns HTTP 409.

---

## Task 3 — Accept and persist `rationale` on `POST /keep` and `POST /discard`

**Files:** `src/api_server.py`, `src/autoresearch_worker.py`

**What to add:**
1. In `api_server.py`: parse `rationale` from the JSON body of `POST /keep` and `POST /discard` and pass it through to the worker functions.
2. In `autoresearch_worker.py`: accept an optional `rationale: str` argument in `keep_autoresearch_candidate()` and `discard_autoresearch_candidate()`. Include it in the memory event payload when appending to `experiment_memory.jsonl`.

Example keep call: `POST /keep` with body `{"rationale": "val_bpb improved 0.03 vs baseline"}`.

**Acceptance criterion:** After a keep or discard, the most recent entry in `experiment_memory.jsonl` contains a `rationale` key with the value that was sent.

---

## Task 4 — Enrich memory schema

**File:** `src/autoresearch_worker.py`

**What to add:** Extend the payload written to `experiment_memory.jsonl` in every `append_memory_event()` call so that `candidate_run`, `keep`, and `discard` events include:

- `experiment_id` — use the short HEAD commit SHA (already available as `short_head_commit(root)`)
- `code_change_summary` — list of changed files (already available in the worker result as `changed_files`)
- `decision_rationale` — the rationale string from Task 3 (empty string if not provided)
- `failure_tag` — on crash/failed events, a short tag such as `"syntax_error"`, `"timeout"`, `"no_improvement"`, `"unknown"` inferred from the error field

Existing memory fields (`setup`, `state`, `worker`, `experiment`, `results_tsv`, `commit`, `recommended_status`, `description`) should be kept as-is.

**Acceptance criterion:** Run one full experiment loop (even a smoke test with a no-op train.py edit). The resulting `experiment_memory.jsonl` entry for the run contains all four new fields.

---

## Task 5 — Write the manager system prompt

**File:** `src/manager_prompt.py` (new file)

**What to add:** A Python module that exports a single string constant `MANAGER_SYSTEM_PROMPT`. The prompt instructs a Claude/GPT manager to:

1. Call `GET /status` — check current state; if `pending`, wait and retry.
2. Call `GET /memory` — read prior experiments to avoid redundant changes.
3. Decide on one concrete, bounded change to `train.py` (learning rate, batch size, model depth, etc.) not already attempted.
4. Call `POST /run` with a JSON body containing `objective` (one sentence) and `patch` (unified diff or natural-language description of the change for the worker).
5. Wait for the run to complete (poll `GET /status` until state is not `running`).
6. Read returned `metrics` and `recommended_status`.
7. Call `POST /keep` with `rationale` if `recommended_status == "improve"`, otherwise call `POST /discard` with `rationale`.
8. Stop after one iteration (Stage 1 scope).

The prompt must be plain text (no f-string substitution needed). Keep it under 600 words.

**Acceptance criterion:** The file exists, imports cleanly (`python3 -c "from src.manager_prompt import MANAGER_SYSTEM_PROMPT; print(len(MANAGER_SYSTEM_PROMPT))"`), and the string is between 200 and 600 words.

---

## Task 6 — Write the demo launcher script

**File:** `scripts/run_demo.sh` (new file, make executable)

**What to add:** A shell script that:

1. Validates that `AUTORESEARCH_ROOT` env var is set (or uses the default `../../nodes/autoresearch-macos`).
2. Starts the control plane API server in the background: `python3 -m src.main api-server --root $AUTORESEARCH_ROOT &` — saves the PID.
3. Waits up to 5 seconds for `GET /health` to return 200.
4. Runs one manager loop using the Rust claw binary: `claw run --manager --system-prompt src/manager_prompt.py --api-base http://localhost:PORT` (or equivalent CLI invocation based on actual claw flags).
5. On exit (trap), kills the background server.

If the Rust claw invocation is not yet wired to accept `--manager` mode, the script may instead run a minimal Python manager stub that imports `MANAGER_SYSTEM_PROMPT`, calls the API endpoints using `urllib`, and prints results. The stub approach is acceptable for Stage 1.

**Acceptance criterion:** Running `bash scripts/run_demo.sh` from the `claw-code/` directory starts the server, executes one experiment loop end-to-end, prints the keep/discard decision, and exits cleanly (server process is cleaned up).

---

## Task 7 — Smoke test all six tasks together

**File:** `notebooks/autoresearch_control_plane_demo.ipynb` — add a new final cell (or update the existing summary cell)

**What to add:** A cell that runs the complete demo narrative from the plan in sequence:

```
1. GET /status  → show initial state
2. GET /memory  → show prior experiments (empty or not)
3. POST /run    → launch experiment with a small train.py change
4. poll GET /status until state != "running"
5. print metrics and recommended_status
6. POST /keep or /discard with rationale
7. GET /memory  → confirm new entry appeared
```

All steps should succeed without manual intervention. Any failure prints a clear error and stops the cell.

**Acceptance criterion:** Running the notebook top-to-bottom completes without errors and prints `demo complete` at the end.

---

## Implementation order

Tasks 1–4 are pure Python additions to existing files — do them first and commit each one separately. Task 5 is a new file, independent of 1–4. Task 6 depends on Task 5. Task 7 depends on all prior tasks.

Suggested commit messages:
- `feat: add GET /memory endpoint to api_server`
- `feat: add concurrent-run guard (409) on POST /run`
- `feat: persist rationale on keep/discard events`
- `feat: enrich memory schema with experiment_id, code_change_summary, rationale, failure_tag`
- `feat: add manager system prompt module`
- `feat: add demo launcher script`
- `test: end-to-end smoke test in control_plane_demo notebook`
