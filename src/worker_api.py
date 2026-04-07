from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .agent_loop import AgentRunResult, run_agent_task
from .llm_backend import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL
from .task_packet import TaskPacket, load_task_packet, render_worker_prompt

DEFAULT_WORKER_DIR = Path('.port_workers')
DEFAULT_ACCEPTANCE_TIMEOUT_SECONDS = 300
INTERNAL_CHANGED_FILE_PREFIXES = ('.port_sessions/', '.port_workers/')


@dataclass
class WorkerRecord:
    worker_id: str
    root: str
    model: str
    host: str
    state: str
    created_at: str
    updated_at: str
    run_count: int = 0
    last_packet: dict[str, Any] | None = None
    last_result: dict[str, Any] | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'WorkerRecord':
        return cls(
            worker_id=payload['worker_id'],
            root=payload['root'],
            model=payload['model'],
            host=payload['host'],
            state=payload['state'],
            created_at=payload['created_at'],
            updated_at=payload['updated_at'],
            run_count=payload.get('run_count', 0),
            last_packet=payload.get('last_packet'),
            last_result=payload.get('last_result'),
            last_error=payload.get('last_error'),
        )


def workers_dir(directory: Path | None = None) -> Path:
    target = directory or DEFAULT_WORKER_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def create_worker(
    root: Path | str | None = None,
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    directory: Path | None = None,
) -> WorkerRecord:
    now = _iso_now()
    worker = WorkerRecord(
        worker_id=uuid4().hex,
        root=str((Path(root) if root is not None else Path.cwd()).resolve()),
        model=model,
        host=host,
        state='ready',
        created_at=now,
        updated_at=now,
    )
    save_worker(worker, directory=directory)
    return worker


def list_workers(directory: Path | None = None) -> list[WorkerRecord]:
    records: list[WorkerRecord] = []
    for path in sorted(workers_dir(directory).glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        records.append(WorkerRecord.from_dict(payload))
    records.sort(key=lambda worker: worker.updated_at, reverse=True)
    return records


def load_worker(worker_id: str, directory: Path | None = None) -> WorkerRecord:
    path = workers_dir(directory) / f'{worker_id}.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    return WorkerRecord.from_dict(payload)


def save_worker(worker: WorkerRecord, directory: Path | None = None) -> Path:
    path = workers_dir(directory) / f'{worker.worker_id}.json'
    path.write_text(json.dumps(worker.to_dict(), indent=2), encoding='utf-8')
    return path


def run_worker(
    worker_id: str,
    packet_source: TaskPacket | str | Path,
    directory: Path | None = None,
    trace: bool = False,
) -> WorkerRecord:
    worker = load_worker(worker_id, directory=directory)
    if worker.state == 'closed':
        raise RuntimeError(f'Worker {worker_id} is closed.')
    packet = packet_source if isinstance(packet_source, TaskPacket) else load_task_packet(packet_source)
    running_worker = _replace_worker(
        worker,
        state='running',
        updated_at=_iso_now(),
        last_packet=packet.to_dict(),
        last_error=None,
    )
    save_worker(running_worker, directory=directory)

    before_paths = _dirty_paths(Path(running_worker.root))
    try:
        agent_result = run_agent_task(
            render_worker_prompt(packet),
            model=running_worker.model,
            host=running_worker.host,
            root=Path(running_worker.root),
            trace=trace,
        )
        verification_steps = _run_acceptance_tests(Path(running_worker.root), packet.acceptance_tests)
        changed_files = _collect_changed_files(Path(running_worker.root), before_paths, agent_result)
        result = _build_worker_result(running_worker, packet, agent_result, verification_steps, changed_files)
        next_state = 'finished' if result['stop_reason'] == 'completed' else 'failed'
        updated_worker = _replace_worker(
            running_worker,
            state=next_state,
            updated_at=_iso_now(),
            run_count=running_worker.run_count + 1,
            last_result=result,
            last_error=None if next_state == 'finished' else result.get('final_answer') or result['stop_reason'],
        )
    except Exception as error:
        updated_worker = _replace_worker(
            running_worker,
            state='failed',
            updated_at=_iso_now(),
            run_count=running_worker.run_count + 1,
            last_result={
                'worker_id': running_worker.worker_id,
                'state': 'failed',
                'tool_calls': [],
                'tool_trace': [],
                'artifacts': [],
                'changed_files': [],
                'verification': {'acceptance_tests': [], 'tool_runs': {'run_tests': [], 'run_build': []}},
                'final_answer': '',
                'stop_reason': 'error',
                'error': str(error),
            },
            last_error=str(error),
        )
    save_worker(updated_worker, directory=directory)
    return updated_worker


def resume_worker(worker_id: str, directory: Path | None = None, trace: bool = False) -> WorkerRecord:
    worker = load_worker(worker_id, directory=directory)
    if worker.last_packet is None:
        raise RuntimeError(f'Worker {worker_id} has no stored task packet to resume.')
    return run_worker(worker_id, TaskPacket.from_dict(worker.last_packet), directory=directory, trace=trace)


def close_worker(worker_id: str, directory: Path | None = None) -> WorkerRecord:
    worker = load_worker(worker_id, directory=directory)
    updated = _replace_worker(worker, state='closed', updated_at=_iso_now())
    save_worker(updated, directory=directory)
    return updated


def _build_worker_result(
    worker: WorkerRecord,
    packet: TaskPacket,
    agent_result: AgentRunResult,
    verification_steps: list[dict[str, Any]],
    changed_files: list[str],
) -> dict[str, Any]:
    tool_run_summary = {
        'run_tests': [entry for entry in agent_result.tool_trace if entry['name'] == 'run_tests'],
        'run_build': [entry for entry in agent_result.tool_trace if entry['name'] == 'run_build'],
    }
    acceptance_success = all(step['success'] for step in verification_steps)
    artifacts = sorted(
        {
            entry['metadata']['path']
            for entry in agent_result.tool_trace
            if isinstance(entry.get('metadata'), dict) and isinstance(entry['metadata'].get('path'), str)
            and entry['name'] == 'edit_file'
        }
        | set(changed_files)
    )
    stop_reason = 'completed' if acceptance_success else 'verification_failed'
    return {
        'worker_id': worker.worker_id,
        'state': 'finished' if acceptance_success else 'failed',
        'packet': packet.to_dict(),
        'tool_calls': list(agent_result.tool_calls),
        'tool_trace': list(agent_result.tool_trace),
        'artifacts': artifacts,
        'changed_files': changed_files,
        'verification': {
            'acceptance_tests': verification_steps,
            'tool_runs': tool_run_summary,
        },
        'final_answer': agent_result.content,
        'stop_reason': stop_reason,
        'turns': agent_result.turns,
        'trace_events': list(agent_result.trace_events),
    }


def _run_acceptance_tests(root: Path, commands: tuple[str, ...]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for command in commands:
        try:
            result = subprocess.run(
                ['/bin/zsh', '-lc', command],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=DEFAULT_ACCEPTANCE_TIMEOUT_SECONDS,
                check=False,
            )
            combined = '\n'.join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
            steps.append(
                {
                    'command': command,
                    'success': result.returncode == 0,
                    'exit_code': result.returncode,
                    'output': combined,
                }
            )
        except subprocess.TimeoutExpired:
            steps.append(
                {
                    'command': command,
                    'success': False,
                    'exit_code': -1,
                    'output': '',
                    'error': f'Command timed out after {DEFAULT_ACCEPTANCE_TIMEOUT_SECONDS} seconds.',
                }
            )
    return steps


def _collect_changed_files(root: Path, before_paths: set[str], agent_result: AgentRunResult) -> list[str]:
    after_paths = _dirty_paths(root)
    edited_paths = {
        entry['metadata']['path']
        for entry in agent_result.tool_trace
        if entry['name'] == 'edit_file'
        and isinstance(entry.get('metadata'), dict)
        and isinstance(entry['metadata'].get('path'), str)
    }
    changed = (after_paths - before_paths) | edited_paths
    return sorted(
        path for path in changed if not any(path.startswith(prefix) for prefix in INTERNAL_CHANGED_FILE_PREFIXES)
    )


def _dirty_paths(root: Path) -> set[str]:
    status = subprocess.run(
        ['git', 'status', '--porcelain'],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        return set()
    paths: set[str] = set()
    for line in status.stdout.splitlines():
        if len(line) < 4:
            continue
        raw_path = line[3:]
        if ' -> ' in raw_path:
            raw_path = raw_path.split(' -> ', 1)[1]
        paths.add(raw_path.strip('"'))
    return paths


def _replace_worker(worker: WorkerRecord, **changes: Any) -> WorkerRecord:
    payload = worker.to_dict()
    payload.update(changes)
    return WorkerRecord.from_dict(payload)


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
