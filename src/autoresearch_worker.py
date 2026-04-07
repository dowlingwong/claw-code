from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm_backend import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL
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
    return TaskPacket(
        objective=packet.objective,
        scope=(
            'Modify only train.py in the autoresearch repo. You may read program.md, README.md, and train.py for context, '
            'but train.py is the only file you may change. Editing any other path is failure. '
            'Do not edit prepare.py, pyproject.toml, or dependencies. Do not run the full training experiment yourself; '
            'the manager will run it after your code change. Use syntax and git inspection tools when helpful.'
        ),
        repo=str(repo_root),
        branch_policy='Stay on the current branch. Do not create or switch branches.',
        acceptance_tests=(packet.syntax_check_command,),
        commit_policy='Do not commit. The manager records and evaluates the experiment after your edit.',
        reporting_contract=(
            'Return a concise summary of the train.py change, intended effect on val_bpb, and any blockers.'
        ),
        escalation_policy='Stop if a required file outside train.py would need modification or if syntax cannot be preserved.',
    )


def run_autoresearch_packet(
    packet_source: AutoresearchExperimentPacket | str | Path,
    root: str | Path | None = None,
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    trace: bool = False,
) -> dict[str, Any]:
    repo_root = resolve_autoresearch_root(root)
    setup = setup_autoresearch(repo_root, initialize_results=True)
    packet = packet_source if isinstance(packet_source, AutoresearchExperimentPacket) else load_autoresearch_packet(packet_source)
    state = load_autoresearch_state(repo_root)
    base_commit = short_head_commit(repo_root)

    worker = create_worker(root=repo_root, model=model, host=host)
    worker_run = run_worker(
        worker.worker_id,
        render_autoresearch_task_packet(packet, repo_root),
        trace=trace,
    )

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


def keep_autoresearch_candidate(root: str | Path | None = None) -> dict[str, Any]:
    repo_root = resolve_autoresearch_root(root)
    state = load_autoresearch_state(repo_root)
    pending = _require_pending_experiment(state)
    metrics = ExperimentMetrics(**pending.experiment)
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
        'state': updated_state.to_dict(),
        'experiment': pending.experiment,
    }
    append_memory_event(repo_root, 'keep', result)
    return result


def discard_autoresearch_candidate(root: str | Path | None = None) -> dict[str, Any]:
    repo_root = resolve_autoresearch_root(root)
    state = load_autoresearch_state(repo_root)
    pending = _require_pending_experiment(state)
    metrics = ExperimentMetrics(**pending.experiment)
    final_status = 'crash' if not metrics.success or metrics.val_bpb is None else 'discard'
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
        run_result = run_autoresearch_packet(packet, root=repo_root, model=model, host=host, trace=trace)
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
    return AutoresearchState.from_dict(payload)


def save_autoresearch_state(state: AutoresearchState, root: str | Path | None = None) -> Path:
    repo_root = resolve_autoresearch_root(root or state.root)
    path = repo_root / DEFAULT_STATE_FILE
    path.write_text(json.dumps(state.to_dict(), indent=2), encoding='utf-8')
    return path


def append_memory_event(root: str | Path | None, event_type: str, payload: dict[str, Any]) -> Path:
    repo_root = resolve_autoresearch_root(root)
    path = repo_root / DEFAULT_MEMORY_FILE
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'time': _iso_now(), 'event': event_type, 'payload': payload}) + '\n')
    return path


def replace_autoresearch_state(state: AutoresearchState, **changes: Any) -> AutoresearchState:
    payload = state.to_dict()
    payload.update(changes)
    return AutoresearchState.from_dict(payload)


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
