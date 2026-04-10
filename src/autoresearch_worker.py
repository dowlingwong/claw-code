from __future__ import annotations

import json
import re
import subprocess
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm_backend import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL, LLMBackend, OllamaBackend, create_backend
from .task_packet import TaskPacket
from .worker_api import create_worker, run_worker
from .autoresearch_runner import (
    DEFAULT_RESULTS_TSV,
    ExperimentMetrics,
    append_results_row,
    best_recorded_bpb,
    resolve_autoresearch_root,
    run_experiment,
    setup_autoresearch,
    short_head_commit,
)

DEFAULT_AUTORESEARCH_TIMEOUT_SECONDS = 600
DEFAULT_STATE_FILE = '.autoresearch_state.json'
DEFAULT_MEMORY_FILE = 'experiment_memory.jsonl'
AUTORESEARCH_BRANCH_PREFIX = 'autoresearch/'
MEMORY_CRASH_EVENT_TYPES = frozenset({'crash', 'baseline_failed', 'run_skipped', 'retry'})
NON_CODE_CHANGE_MARKERS = ('__pycache__/',)
NON_CODE_CHANGE_SUFFIXES = ('.pyc',)
SUMMARY_EVENT_TYPES = frozenset({'candidate_run', 'keep', 'discard', 'crash'})


@dataclass(frozen=True)
class AutoresearchExperimentPacket:
    objective: str
    description: str
    train_command: str = 'uv run train.py > run.log 2>&1'
    timeout_seconds: int = DEFAULT_AUTORESEARCH_TIMEOUT_SECONDS
    log_path: str = 'run.log'
    results_tsv: str = DEFAULT_RESULTS_TSV
    syntax_check_command: str = 'python3 -m py_compile train.py'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'AutoresearchExperimentPacket':
        return cls(
            objective=str(payload.get('objective', '')),
            description=str(payload.get('description', '')),
            train_command=str(payload.get('train_command', 'uv run train.py > run.log 2>&1')),
            timeout_seconds=int(payload.get('timeout_seconds', DEFAULT_AUTORESEARCH_TIMEOUT_SECONDS)),
            log_path=str(payload.get('log_path', 'run.log')),
            results_tsv=str(payload.get('results_tsv', DEFAULT_RESULTS_TSV)),
            syntax_check_command=str(payload.get('syntax_check_command', 'python3 -m py_compile train.py')),
        )


@dataclass(frozen=True)
class PendingExperiment:
    commit: str
    base_commit: str
    description: str
    packet: dict[str, Any]
    worker: dict[str, Any]
    experiment: dict[str, Any]
    recommended_status: str
    results_tsv: str
    log_path: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'PendingExperiment':
        return cls(
            commit=payload['commit'],
            base_commit=payload['base_commit'],
            description=payload['description'],
            packet=payload['packet'],
            worker=payload['worker'],
            experiment=payload['experiment'],
            recommended_status=payload['recommended_status'],
            results_tsv=payload['results_tsv'],
            log_path=payload['log_path'],
            created_at=payload['created_at'],
        )


@dataclass(frozen=True)
class AutoresearchState:
    root: str
    branch: str
    baseline_commit: str | None = None
    baseline_bpb: float | None = None
    best_commit: str | None = None
    best_bpb: float | None = None
    pending_experiment: dict[str, Any] | None = None
    last_decision: str | None = None
    updated_at: str = ''

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'AutoresearchState':
        return cls(
            root=payload['root'],
            branch=payload['branch'],
            baseline_commit=payload.get('baseline_commit'),
            baseline_bpb=payload.get('baseline_bpb'),
            best_commit=payload.get('best_commit'),
            best_bpb=payload.get('best_bpb'),
            pending_experiment=payload.get('pending_experiment'),
            last_decision=payload.get('last_decision'),
            updated_at=payload.get('updated_at', ''),
        )


def load_autoresearch_packet(path: str | Path) -> AutoresearchExperimentPacket:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('autoresearch packet must be a JSON object')
    packet = AutoresearchExperimentPacket.from_dict(payload)
    validate_autoresearch_packet(packet)
    return packet


def validate_autoresearch_packet(packet: AutoresearchExperimentPacket) -> None:
    errors: list[str] = []
    if not packet.objective.strip():
        errors.append('objective must not be empty')
    if not packet.description.strip():
        errors.append('description must not be empty')
    if packet.timeout_seconds < 1:
        errors.append('timeout_seconds must be >= 1')
    if not packet.train_command.strip():
        errors.append('train_command must not be empty')
    if errors:
        raise ValueError('; '.join(errors))


def render_autoresearch_task_packet(packet: AutoresearchExperimentPacket, repo_root: Path) -> TaskPacket:
    objective = packet.objective.strip()
    if packet.description.strip():
        objective = f'{objective}\n\nExperiment description: {packet.description.strip()}'
    return TaskPacket(
        objective=objective,
        scope=(
            'Modify only train.py in the autoresearch repo. You may read program.md, README.md, and train.py for context, '
            'but train.py is the only file you may change. Editing any other path is failure. '
            'Do not edit prepare.py, pyproject.toml, or dependencies. Do not run the full training experiment yourself; '
            'the manager will run it after your code change. '
            'For simple parameter changes, prefer an exact single-line replacement of the existing assignment in train.py. '
            'Preserve valid Python syntax and do not append extra punctuation after numeric literals. '
            'Use syntax and git inspection tools when helpful.'
        ),
        repo=str(repo_root),
        branch_policy='Stay on the current branch. Do not create or switch branches.',
        acceptance_tests=(packet.syntax_check_command,),
        commit_policy='Do not commit. The manager records and evaluates the experiment after your edit.',
        reporting_contract=(
            'Return a concise summary of the exact train.py line you changed, the intended effect on val_bpb, and any blockers.'
        ),
        escalation_policy='Stop if a required file outside train.py would need modification or if syntax cannot be preserved.',
    )


def run_autoresearch_packet(
    packet_source: AutoresearchExperimentPacket | str | Path,
    root: str | Path | None = None,
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    trace: bool = False,
    backend: LLMBackend | None = None,
) -> dict[str, Any]:
    repo_root = resolve_autoresearch_root(root)
    setup = setup_autoresearch(repo_root, initialize_results=True)
    packet = packet_source if isinstance(packet_source, AutoresearchExperimentPacket) else load_autoresearch_packet(packet_source)
    state = load_autoresearch_state(repo_root)
    base_commit = short_head_commit(repo_root)

    resolved_backend = backend or OllamaBackend(model=model, host=host)
    worker = create_worker(root=repo_root, model=resolved_backend.model, host=resolved_backend.host)
    worker_run = run_worker(
        worker.worker_id,
        render_autoresearch_task_packet(packet, repo_root),
        trace=trace,
        backend=resolved_backend,
    )

    worker_run = _repair_or_fallback_worker_edit(repo_root, packet, worker_run)
    changed_files = _project_changed_files(worker_run.last_result.get('changed_files', []) if worker_run.last_result else [])
    if 'train.py' not in changed_files:
        result = {
            'setup': setup.to_dict(),
            'state': state.to_dict(),
            'worker': worker_run.to_dict(),
            'experiment': None,
            'results_tsv': str((repo_root / packet.results_tsv).resolve()),
            'recommended_status': 'discard',
            'error': 'worker did not modify train.py',
        }
        append_memory_event(repo_root, 'run_skipped', result)
        return result

    commit = _commit_train_change(repo_root, packet.description)
    experiment = run_experiment(
        root=repo_root,
        command=packet.train_command,
        log_path=packet.log_path,
        timeout_seconds=packet.timeout_seconds,
    )
    previous_best = state.best_bpb if state.best_bpb is not None else best_recorded_bpb(repo_root, packet.results_tsv)
    recommended_status = _recommend_status(experiment, previous_best)
    pending = PendingExperiment(
        commit=commit,
        base_commit=base_commit,
        description=packet.description,
        packet=packet.to_dict(),
        worker=worker_run.to_dict(),
        experiment=experiment.to_dict(),
        recommended_status=recommended_status,
        results_tsv=packet.results_tsv,
        log_path=packet.log_path,
        created_at=_iso_now(),
    )
    updated_state = replace_autoresearch_state(
        state,
        branch=_current_branch(repo_root),
        pending_experiment=pending.to_dict(),
        updated_at=_iso_now(),
    )
    save_autoresearch_state(updated_state, repo_root)

    result = {
        'setup': setup.to_dict(),
        'state': updated_state.to_dict(),
        'worker': worker_run.to_dict(),
        'experiment': experiment.to_dict(),
        'results_tsv': str((repo_root / packet.results_tsv).resolve()),
        'commit': commit,
        'recommended_status': recommended_status,
        'description': packet.description,
    }
    append_memory_event(repo_root, 'candidate_run', result)
    return result


def ensure_autoresearch_baseline(
    root: str | Path | None = None,
    command: str = 'uv run train.py > run.log 2>&1',
    log_path: str = 'run.log',
    results_tsv: str = DEFAULT_RESULTS_TSV,
    timeout_seconds: int = DEFAULT_AUTORESEARCH_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    repo_root = resolve_autoresearch_root(root)
    setup = setup_autoresearch(repo_root, initialize_results=True)
    state = load_autoresearch_state(repo_root)
    existing_best = state.best_bpb if state.best_bpb is not None else best_recorded_bpb(repo_root, results_tsv)
    if existing_best is not None:
        if state.best_bpb is None:
            state = replace_autoresearch_state(
                state,
                branch=_current_branch(repo_root),
                best_bpb=existing_best,
                updated_at=_iso_now(),
            )
            save_autoresearch_state(state, repo_root)
        result = {
            'setup': setup.to_dict(),
            'state': state.to_dict(),
            'baseline_created': False,
            'reason': 'baseline already recorded',
        }
        append_memory_event(repo_root, 'baseline_skipped', result)
        return result

    commit = short_head_commit(repo_root)
    experiment = run_experiment(repo_root, command=command, log_path=log_path, timeout_seconds=timeout_seconds)
    if not experiment.success or experiment.val_bpb is None:
        result = {
            'setup': setup.to_dict(),
            'state': state.to_dict(),
            'baseline_created': False,
            'experiment': experiment.to_dict(),
            'error': 'baseline experiment failed',
        }
        append_memory_event(repo_root, 'baseline_failed', result)
        return result

    append_results_row(repo_root, commit=commit, metrics=experiment, status='keep', description='baseline', results_path=results_tsv)
    updated_state = replace_autoresearch_state(
        state,
        branch=_current_branch(repo_root),
        baseline_commit=commit,
        baseline_bpb=experiment.val_bpb,
        best_commit=commit,
        best_bpb=experiment.val_bpb,
        pending_experiment=None,
        last_decision='baseline',
        updated_at=_iso_now(),
    )
    save_autoresearch_state(updated_state, repo_root)
    result = {
        'setup': setup.to_dict(),
        'state': updated_state.to_dict(),
        'baseline_created': True,
        'experiment': experiment.to_dict(),
        'commit': commit,
    }
    append_memory_event(repo_root, 'baseline_created', result)
    return result


def keep_autoresearch_candidate(root: str | Path | None = None, rationale: str = '') -> dict[str, Any]:
    repo_root = resolve_autoresearch_root(root)
    state = load_autoresearch_state(repo_root)
    pending = _require_pending_experiment(state)
    metrics = ExperimentMetrics(**pending.experiment)
    change_summary = _pending_change_summary(pending)
    append_results_row(
        repo_root,
        commit=pending.commit,
        metrics=metrics,
        status='keep',
        description=pending.description,
        results_path=pending.results_tsv,
    )
    improved = metrics.val_bpb is not None and (state.best_bpb is None or metrics.val_bpb < state.best_bpb)
    updated_state = replace_autoresearch_state(
        state,
        branch=_current_branch(repo_root),
        best_commit=pending.commit if improved else state.best_commit,
        best_bpb=metrics.val_bpb if improved else state.best_bpb,
        pending_experiment=None,
        last_decision='keep',
        updated_at=_iso_now(),
    )
    save_autoresearch_state(updated_state, repo_root)
    result = {
        'decision': 'keep',
        'commit': pending.commit,
        'improved_best': improved,
        'description': pending.description,
        'code_change_summary': change_summary,
        'rationale': rationale,
        'state': updated_state.to_dict(),
        'experiment': pending.experiment,
    }
    append_memory_event(repo_root, 'keep', result)
    return result


def discard_autoresearch_candidate(root: str | Path | None = None, rationale: str = '') -> dict[str, Any]:
    repo_root = resolve_autoresearch_root(root)
    state = load_autoresearch_state(repo_root)
    pending = _require_pending_experiment(state)
    metrics = ExperimentMetrics(**pending.experiment)
    final_status = 'crash' if not metrics.success or metrics.val_bpb is None else 'discard'
    change_summary = _pending_change_summary(pending)
    _git(repo_root, ['git', 'reset', '--hard', pending.base_commit], check=False)
    append_results_row(
        repo_root,
        commit=pending.commit,
        metrics=metrics,
        status=final_status,
        description=pending.description,
        results_path=pending.results_tsv,
    )
    updated_state = replace_autoresearch_state(
        state,
        branch=_current_branch(repo_root),
        pending_experiment=None,
        last_decision=final_status,
        updated_at=_iso_now(),
    )
    save_autoresearch_state(updated_state, repo_root)
    result = {
        'decision': final_status,
        'reverted_to_commit': pending.base_commit,
        'current_commit': short_head_commit(repo_root),
        'commit': pending.commit,
        'description': pending.description,
        'code_change_summary': change_summary,
        'rationale': rationale,
        'state': updated_state.to_dict(),
        'experiment': pending.experiment,
    }
    append_memory_event(repo_root, final_status, result)
    return result


def retry_autoresearch_candidate(root: str | Path | None = None) -> dict[str, Any]:
    repo_root = resolve_autoresearch_root(root)
    state = load_autoresearch_state(repo_root)
    pending = _require_pending_experiment(state)
    packet = AutoresearchExperimentPacket.from_dict(pending.packet)
    experiment = run_experiment(
        root=repo_root,
        command=packet.train_command,
        log_path=packet.log_path,
        timeout_seconds=packet.timeout_seconds,
    )
    previous_best = state.best_bpb if state.best_bpb is not None else best_recorded_bpb(repo_root, packet.results_tsv)
    recommended_status = _recommend_status(experiment, previous_best)
    refreshed = PendingExperiment(
        commit=pending.commit,
        base_commit=pending.base_commit,
        description=pending.description,
        packet=pending.packet,
        worker=pending.worker,
        experiment=experiment.to_dict(),
        recommended_status=recommended_status,
        results_tsv=pending.results_tsv,
        log_path=pending.log_path,
        created_at=pending.created_at,
    )
    updated_state = replace_autoresearch_state(
        state,
        pending_experiment=refreshed.to_dict(),
        updated_at=_iso_now(),
    )
    save_autoresearch_state(updated_state, repo_root)
    result = {
        'state': updated_state.to_dict(),
        'experiment': experiment.to_dict(),
        'recommended_status': recommended_status,
        'commit': pending.commit,
        'description': pending.description,
    }
    append_memory_event(repo_root, 'retry', result)
    return result


def loop_autoresearch(
    packet_source: AutoresearchExperimentPacket | str | Path,
    root: str | Path | None = None,
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    iterations: int = 1,
    retry_limit: int = 1,
    require_isolated_branch: bool = True,
    trace: bool = False,
    backend: LLMBackend | None = None,
) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError('iterations must be >= 1')
    repo_root = resolve_autoresearch_root(root)
    if require_isolated_branch and not _is_isolated_branch(_current_branch(repo_root)):
        raise RuntimeError(
            'autoresearch loop requires an isolated branch named autoresearch/<tag>. '
            'Run `python3 -m src.main autoresearch isolate --root ... --branch autoresearch/<tag> --create` first, '
            'or pass --allow-any-branch for a local smoke test.'
        )
    packet = packet_source if isinstance(packet_source, AutoresearchExperimentPacket) else load_autoresearch_packet(packet_source)
    setup = setup_autoresearch(repo_root, initialize_results=True)
    baseline = ensure_autoresearch_baseline(repo_root, results_tsv=packet.results_tsv)
    history: list[dict[str, Any]] = []

    for iteration in range(1, iterations + 1):
        run_result = run_autoresearch_packet(packet, root=repo_root, model=model, host=host, trace=trace, backend=backend)
        retries = 0
        while run_result.get('recommended_status') == 'crash' and retries < retry_limit:
            retries += 1
            run_result = retry_autoresearch_candidate(repo_root)
            run_result['retry_count'] = retries

        state = load_autoresearch_state(repo_root)
        if state.pending_experiment is None:
            history.append(
                {
                    'iteration': iteration,
                    'run': run_result,
                    'decision': {
                        'decision': 'no_pending_experiment',
                        'state': state.to_dict(),
                    },
                }
            )
            continue

        decision = (
            keep_autoresearch_candidate(repo_root)
            if run_result.get('recommended_status') == 'keep'
            else discard_autoresearch_candidate(repo_root)
        )
        history.append(
            {
                'iteration': iteration,
                'run': run_result,
                'decision': decision,
            }
        )

    result = {
        'setup': setup.to_dict(),
        'baseline': baseline,
        'state': load_autoresearch_state(repo_root).to_dict(),
        'history': history,
    }
    append_memory_event(repo_root, 'loop_complete', {'iterations': iterations, 'history_size': len(history)})
    return result


def load_autoresearch_state(root: str | Path | None = None) -> AutoresearchState:
    repo_root = resolve_autoresearch_root(root)
    path = repo_root / DEFAULT_STATE_FILE
    if not path.exists():
        state = AutoresearchState(
            root=str(repo_root),
            branch=_current_branch(repo_root),
            updated_at=_iso_now(),
        )
        save_autoresearch_state(state, repo_root)
        return state
    payload = json.loads(path.read_text(encoding='utf-8'))
    state = AutoresearchState.from_dict(payload)
    # Recover from a state file left with a stale pending_experiment (e.g. the
    # process crashed between git-reset and state-save during a discard).
    state = _recover_stale_pending(state, repo_root)
    return state


def _recover_stale_pending(state: AutoresearchState, repo_root: Path) -> AutoresearchState:
    """Clear pending_experiment if its commit no longer exists in git history.

    This handles the case where a `discard` git-reset ran but the subsequent
    state-file write was interrupted, leaving state.json pointing at a commit
    that was rolled back.
    """
    if state.pending_experiment is None:
        return state
    try:
        pending = PendingExperiment.from_dict(state.pending_experiment)
    except (KeyError, TypeError):
        # Malformed pending block — clear it defensively.
        return replace_autoresearch_state(state, pending_experiment=None, updated_at=_iso_now())
    commit = pending.commit.replace('-dirty', '').replace('-uncommitted', '')
    check = _git(repo_root, ['git', 'cat-file', '-e', f'{commit}^{{commit}}'], check=False)
    if check.returncode != 0:
        # Commit is gone (was rolled back). Rebuild best_bpb from TSV.
        recovered_best = best_recorded_bpb(repo_root, pending.results_tsv)
        recovered = replace_autoresearch_state(
            state,
            pending_experiment=None,
            best_bpb=recovered_best if recovered_best is not None else state.best_bpb,
            last_decision='recovered',
            updated_at=_iso_now(),
        )
        save_autoresearch_state(recovered, repo_root)
        return recovered
    return state


def save_autoresearch_state(state: AutoresearchState, root: str | Path | None = None) -> Path:
    repo_root = resolve_autoresearch_root(root or state.root)
    path = repo_root / DEFAULT_STATE_FILE
    path.write_text(json.dumps(state.to_dict(), indent=2), encoding='utf-8')
    return path


def append_memory_event(root: str | Path | None, event_type: str, payload: dict[str, Any]) -> Path:
    repo_root = resolve_autoresearch_root(root)
    path = repo_root / DEFAULT_MEMORY_FILE
    enriched_payload = _enrich_memory_payload(repo_root, event_type, payload)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'time': _iso_now(), 'event': event_type, 'payload': enriched_payload}) + '\n')
    return path


def replace_autoresearch_state(state: AutoresearchState, **changes: Any) -> AutoresearchState:
    payload = state.to_dict()
    payload.update(changes)
    return AutoresearchState.from_dict(payload)


_SIMPLE_ASSIGNMENT_PATTERNS = (
    re.compile(
        r'(?:set|setting)\s+`?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>\[[^\]]+\]|[-+0-9.eE]+|True|False|None|".*?"|\'.*?\')`?',
        flags=re.IGNORECASE,
    ),
    re.compile(
        r'replace\s+`?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^`]+`?\s+with\s+`?(?P<value>\[[^\]]+\]|[-+0-9.eE]+|True|False|None|".*?"|\'.*?\')`?',
        flags=re.IGNORECASE,
    ),
)


def _repair_or_fallback_worker_edit(
    repo_root: Path,
    packet: AutoresearchExperimentPacket,
    worker_run: Any,
) -> Any:
    last_result = worker_run.last_result or {}
    changed_files = _project_changed_files(last_result.get('changed_files', []))
    verification_steps = list(last_result.get('verification', {}).get('acceptance_tests', []))
    syntax_ok = all(step.get('success') for step in verification_steps) if verification_steps else False
    needs_fallback = 'train.py' not in changed_files or not syntax_ok
    if not needs_fallback:
        return worker_run

    fallback = _extract_simple_train_assignment(packet)
    if fallback is None:
        return worker_run

    fallback_applied = _apply_train_assignment(repo_root / 'train.py', fallback['name'], fallback['value'])
    if not fallback_applied:
        return worker_run

    syntax_check = _run_syntax_check(repo_root, packet.syntax_check_command)
    if not syntax_check.get('success'):
        return worker_run

    payload = worker_run.to_dict()
    last_result_payload = deepcopy(last_result)
    fallback_trace = {
        'turn': 0,
        'name': 'deterministic_assignment_fallback',
        'arguments': {'path': 'train.py', 'name': fallback['name'], 'value': fallback['value']},
        'success': True,
        'output': f"Updated train.py via deterministic fallback: {fallback['name']} = {fallback['value']}",
        'metadata': {'path': 'train.py', 'deterministic_fallback': True},
        'error': None,
    }
    tool_trace = list(last_result_payload.get('tool_trace', []))
    tool_trace.append(fallback_trace)
    verification_payload = deepcopy(last_result_payload.get('verification', {'acceptance_tests': [], 'tool_runs': {'run_tests': [], 'run_build': []}}))
    verification_payload['acceptance_tests'] = [syntax_check]
    last_result_payload['tool_trace'] = tool_trace
    last_result_payload['changed_files'] = ['train.py']
    last_result_payload['artifacts'] = sorted(set(last_result_payload.get('artifacts', [])) | {'train.py'})
    last_result_payload['verification'] = verification_payload
    last_result_payload['state'] = 'finished'
    last_result_payload['stop_reason'] = 'completed'
    final_answer = (last_result_payload.get('final_answer') or '').strip()
    fallback_note = f"Deterministic fallback applied: set {fallback['name']} = {fallback['value']} in train.py."
    last_result_payload['final_answer'] = f'{final_answer}\n{fallback_note}'.strip()
    payload['last_result'] = last_result_payload
    payload['state'] = 'finished'
    payload['last_error'] = None
    return worker_run.__class__.from_dict(payload)


def _extract_simple_train_assignment(packet: AutoresearchExperimentPacket) -> dict[str, str] | None:
    search_text = '\n'.join(part for part in (packet.objective, packet.description) if part.strip())
    for pattern in _SIMPLE_ASSIGNMENT_PATTERNS:
        match = pattern.search(search_text)
        if match:
            return {'name': match.group('name'), 'value': _normalize_assignment_value(match.group('value'))}
    return None


def _normalize_assignment_value(raw: str) -> str:
    return raw.strip().rstrip('.')


def _apply_train_assignment(train_path: Path, name: str, value: str) -> bool:
    if not train_path.exists():
        return False
    text = train_path.read_text(encoding='utf-8')
    pattern = re.compile(rf'^(?P<indent>\s*){re.escape(name)}\s*=.*$', flags=re.MULTILINE)
    replacement = f'{name} = {value}'
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        return False
    train_path.write_text(updated, encoding='utf-8')
    return True


def _run_syntax_check(repo_root: Path, command: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ['/bin/zsh', '-lc', command],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            'command': command,
            'success': False,
            'exit_code': -1,
            'output': '',
            'error': 'Command timed out after 300 seconds.',
        }
    combined = '\n'.join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    return {
        'command': command,
        'success': result.returncode == 0,
        'exit_code': result.returncode,
        'output': combined,
    }


def autoresearch_status(root: str | Path | None = None) -> dict[str, Any]:
    repo_root = resolve_autoresearch_root(root)
    state = load_autoresearch_state(repo_root)
    setup = setup_autoresearch(repo_root, initialize_results=True)
    memory_path = repo_root / DEFAULT_MEMORY_FILE
    results_path = repo_root / DEFAULT_RESULTS_TSV
    return {
        'setup': setup.to_dict(),
        'state': state.to_dict(),
        'current_branch': _current_branch(repo_root),
        'isolated_branch': _is_isolated_branch(_current_branch(repo_root)),
        'paths': {
            'state_file': str((repo_root / DEFAULT_STATE_FILE).resolve()),
            'memory_file': str(memory_path.resolve()),
            'results_tsv': str(results_path.resolve()),
        },
        'memory_event_count': _count_memory_events(memory_path),
        'has_pending_experiment': state.pending_experiment is not None,
    }


def ensure_autoresearch_branch(
    root: str | Path | None = None,
    branch: str | None = None,
    create: bool = False,
    from_ref: str = 'HEAD',
) -> dict[str, Any]:
    repo_root = resolve_autoresearch_root(root)
    previous_branch = _current_branch(repo_root)
    target_branch = branch or (previous_branch if _is_isolated_branch(previous_branch) else _suggest_branch_name())
    if not _is_isolated_branch(target_branch):
        raise ValueError(f'branch must start with {AUTORESEARCH_BRANCH_PREFIX}')
    created = False
    if previous_branch != target_branch:
        verify = _git(repo_root, ['git', 'rev-parse', '--verify', target_branch], check=False)
        if verify.returncode == 0:
            _git(repo_root, ['git', 'checkout', target_branch], check=True)
        elif create:
            _git(repo_root, ['git', 'checkout', '-b', target_branch, from_ref], check=True)
            created = True
        else:
            raise RuntimeError(
                f'Branch {target_branch} does not exist. Re-run with create=True or use '
                '`python3 -m src.main autoresearch isolate --create`.'
            )
    state = load_autoresearch_state(repo_root)
    updated_state = replace_autoresearch_state(
        state,
        branch=_current_branch(repo_root),
        updated_at=_iso_now(),
    )
    save_autoresearch_state(updated_state, repo_root)
    result = {
        'root': str(repo_root),
        'previous_branch': previous_branch,
        'current_branch': _current_branch(repo_root),
        'isolated_branch': True,
        'created': created,
        'suggested_branch': _suggest_branch_name(),
    }
    append_memory_event(repo_root, 'branch_isolated', result)
    return result


def load_memory_events(root: str | Path | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    repo_root = resolve_autoresearch_root(root)
    path = repo_root / DEFAULT_MEMORY_FILE
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)

    if limit == 0:
        return []
    if limit is not None and limit > 0:
        return entries[-limit:]
    return entries


def summarize_memory_events(root: str | Path | None = None, limit: int | None = 20) -> dict[str, Any]:
    events = load_memory_events(root, limit=None)
    recent_source = events[-limit:] if limit is not None and limit > 0 else events
    counts_by_event: dict[str, int] = {}
    idea_rollup: dict[str, dict[str, Any]] = {}
    recent_events: list[dict[str, Any]] = []
    latest_decision: dict[str, Any] | None = None

    for entry in events:
        event_name = str(entry.get('event', 'unknown'))
        counts_by_event[event_name] = counts_by_event.get(event_name, 0) + 1
        payload = entry.get('payload', {})
        if not isinstance(payload, dict):
            continue
        idea_key = str(payload.get('idea_key') or '')
        if idea_key:
            rollup = idea_rollup.setdefault(
                idea_key,
                {
                    'idea_key': idea_key,
                    'latest_description': '',
                    'attempts': 0,
                    'keeps': 0,
                    'discards': 0,
                    'crashes': 0,
                    'latest_outcome': '',
                    'latest_experiment_id': '',
                },
            )
            description = payload.get('description')
            if isinstance(description, str) and description:
                rollup['latest_description'] = description
            if event_name in SUMMARY_EVENT_TYPES:
                rollup['attempts'] += 1 if event_name == 'candidate_run' else 0
            if event_name == 'keep':
                rollup['keeps'] += 1
                rollup['latest_outcome'] = 'keep'
            elif event_name == 'discard':
                rollup['discards'] += 1
                rollup['latest_outcome'] = 'discard'
            elif event_name == 'crash':
                rollup['crashes'] += 1
                rollup['latest_outcome'] = 'crash'
            elif event_name == 'candidate_run' and not rollup['latest_outcome']:
                rollup['latest_outcome'] = str(payload.get('recommended_status') or '')
            experiment_id = payload.get('experiment_id')
            if isinstance(experiment_id, str) and experiment_id:
                rollup['latest_experiment_id'] = experiment_id

        if event_name in {'keep', 'discard', 'crash'}:
            latest_decision = {
                'time': entry.get('time'),
                'event': event_name,
                'experiment_id': payload.get('experiment_id'),
                'idea_key': payload.get('idea_key', ''),
                'decision_rationale': payload.get('decision_rationale', ''),
                'failure_tag': payload.get('failure_tag', ''),
            }

    for entry in recent_source:
        payload = entry.get('payload', {})
        if not isinstance(payload, dict):
            payload = {}
        recent_events.append(
            {
                'time': entry.get('time'),
                'event': entry.get('event'),
                'experiment_id': payload.get('experiment_id', ''),
                'idea_key': payload.get('idea_key', ''),
                'description': payload.get('description', ''),
                'decision_rationale': payload.get('decision_rationale', ''),
                'failure_tag': payload.get('failure_tag', ''),
                'recommended_status': payload.get('recommended_status', ''),
                'val_bpb': _extract_val_bpb(payload),
            }
        )

    idea_rollup_list = sorted(
        idea_rollup.values(),
        key=lambda item: (-(item['keeps']), -(item['attempts']), item['idea_key']),
    )
    return {
        'recent_events': recent_events,
        'counts_by_event': counts_by_event,
        'idea_rollup': idea_rollup_list,
        'latest_decision': latest_decision,
        'memory_event_count': len(events),
    }


def _project_changed_files(paths: list[str]) -> list[str]:
    return sorted(path for path in paths if not path.startswith('.port_'))


def _recommend_status(metrics: ExperimentMetrics, previous_best: float | None) -> str:
    if not metrics.success or metrics.val_bpb is None:
        return 'crash'
    if previous_best is None or metrics.val_bpb < previous_best:
        return 'keep'
    return 'discard'


def _commit_train_change(root: Path, description: str) -> str:
    message = f'autoresearch: {description.strip()}'
    subprocess.run(['git', 'add', '--', 'train.py'], cwd=root, capture_output=True, text=True, check=False)
    commit = subprocess.run(
        ['git', 'commit', '-m', message[:120]],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode == 0:
        return short_head_commit(root)
    head = short_head_commit(root)
    return f'{head}-dirty' if head != 'unknown' else 'uncommitted'


def _require_pending_experiment(state: AutoresearchState) -> PendingExperiment:
    if state.pending_experiment is None:
        raise RuntimeError('No pending autoresearch experiment is recorded.')
    return PendingExperiment.from_dict(state.pending_experiment)


def _pending_change_summary(pending: PendingExperiment) -> list[str]:
    worker = pending.worker if isinstance(pending.worker, dict) else {}
    last_result = worker.get('last_result') if isinstance(worker, dict) else {}
    if not isinstance(last_result, dict):
        return []
    changed_files = last_result.get('changed_files', [])
    if not isinstance(changed_files, list):
        return []
    return _normalize_code_change_summary(changed_files)


def _enrich_memory_payload(repo_root: Path, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(payload)
    experiment_id = _infer_experiment_id(repo_root, payload)
    code_change_summary = _infer_code_change_summary(payload)
    rationale = _infer_decision_rationale(payload)
    failure_tag = _infer_failure_tag(event_type, payload)
    idea_key = _infer_idea_key(payload)
    val_bpb = _extract_val_bpb(payload)

    if event_type in {'candidate_run', 'keep', 'discard', 'crash'}:
        enriched.setdefault('experiment_id', experiment_id)
        enriched.setdefault('code_change_summary', code_change_summary)
        enriched.setdefault('decision_rationale', rationale)
        enriched.setdefault('failure_tag', failure_tag)
        enriched.setdefault('idea_key', idea_key)
        enriched.setdefault('decision', event_type)
        enriched.setdefault('val_bpb', val_bpb)
    elif event_type in MEMORY_CRASH_EVENT_TYPES:
        enriched.setdefault('failure_tag', failure_tag)

    return enriched


def _infer_experiment_id(repo_root: Path, payload: dict[str, Any]) -> str:
    commit = payload.get('commit')
    if isinstance(commit, str) and commit:
        return commit

    worker = payload.get('worker')
    if isinstance(worker, dict):
        nested_commit = worker.get('commit')
        if isinstance(nested_commit, str) and nested_commit:
            return nested_commit

    return short_head_commit(repo_root)


def _infer_code_change_summary(payload: dict[str, Any]) -> list[str]:
    summary = payload.get('code_change_summary')
    if isinstance(summary, list):
        return _normalize_code_change_summary(summary)

    worker = payload.get('worker')
    if isinstance(worker, dict):
        last_result = worker.get('last_result')
        if isinstance(last_result, dict):
            changed_files = last_result.get('changed_files', [])
            if isinstance(changed_files, list):
                return _normalize_code_change_summary(changed_files)

    return []


def _normalize_code_change_summary(paths: list[Any]) -> list[str]:
    normalized: list[str] = []
    for raw_path in paths:
        if not isinstance(raw_path, str):
            continue
        path = raw_path.strip()
        if not path:
            continue
        if any(marker in path for marker in NON_CODE_CHANGE_MARKERS):
            continue
        if path.endswith(NON_CODE_CHANGE_SUFFIXES):
            continue
        normalized.append(path)
    return sorted(dict.fromkeys(normalized))


def _infer_decision_rationale(payload: dict[str, Any]) -> str:
    rationale = payload.get('decision_rationale')
    if isinstance(rationale, str):
        return rationale
    rationale = payload.get('rationale')
    if isinstance(rationale, str):
        return rationale
    return ''


def _infer_idea_key(payload: dict[str, Any]) -> str:
    for key in ('description', 'objective'):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            lowered = value.lower().strip()
            normalized = re.sub(r'[^a-z0-9]+', '-', lowered).strip('-')
            return normalized[:80]

    packet = payload.get('packet')
    if isinstance(packet, dict):
        for key in ('description', 'objective'):
            value = packet.get(key)
            if isinstance(value, str) and value.strip():
                lowered = value.lower().strip()
                normalized = re.sub(r'[^a-z0-9]+', '-', lowered).strip('-')
                return normalized[:80]
    return ''


def _extract_val_bpb(payload: dict[str, Any]) -> float | None:
    value = payload.get('val_bpb')
    if isinstance(value, (int, float)):
        return float(value)
    experiment = payload.get('experiment')
    if isinstance(experiment, dict):
        nested = experiment.get('val_bpb')
        if isinstance(nested, (int, float)):
            return float(nested)
    return None


def _infer_failure_tag(event_type: str, payload: dict[str, Any]) -> str:
    if event_type == 'discard':
        return 'no_improvement'

    experiment = payload.get('experiment')
    if isinstance(experiment, dict):
        if experiment.get('timed_out'):
            return 'timeout'
        error = experiment.get('error')
        tail = experiment.get('tail')
        tag = _failure_tag_from_text(error, tail)
        if tag:
            return tag
        if experiment.get('success') is False:
            return 'unknown'

    error = payload.get('error')
    if isinstance(error, str) and error:
        if 'did not modify train.py' in error:
            return 'no_code_change'
        tag = _failure_tag_from_text(error)
        if tag:
            return tag

    if event_type == 'crash':
        return 'unknown'
    return ''


def _failure_tag_from_text(*parts: Any) -> str:
    text = '\n'.join(str(part) for part in parts if isinstance(part, str) and part).lower()
    if not text:
        return ''
    patterns = (
        ('timeout', r'timed?\s*out|timeout'),
        ('syntax_error', r'syntaxerror|indentationerror'),
        ('oom', r'out of memory|oom|cuda out of memory|mps'),
        ('assertion_failed', r'assertionerror'),
        ('import_error', r'importerror|modulenotfounderror'),
        ('runtime_error', r'runtimeerror'),
        ('value_error', r'valueerror'),
        ('type_error', r'typeerror'),
    )
    for tag, pattern in patterns:
        if re.search(pattern, text):
            return tag
    return ''


def _current_branch(root: Path) -> str:
    result = _git(root, ['git', 'branch', '--show-current'], check=False)
    branch = result.stdout.strip()
    return branch or 'DETACHED'


def _git(root: Path, args: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=root, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f'git command failed: {" ".join(args)}')
    return result


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _count_memory_events(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding='utf-8').splitlines() if line.strip())


def _suggest_branch_name() -> str:
    return f'{AUTORESEARCH_BRANCH_PREFIX}{datetime.now().strftime("%b%d").lower()}'


def _is_isolated_branch(branch: str) -> bool:
    return branch.startswith(AUTORESEARCH_BRANCH_PREFIX)
