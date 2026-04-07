from __future__ import annotations

import csv
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

RESULTS_HEADER = 'commit\tval_bpb\tmemory_gb\tstatus\tdescription\n'
DEFAULT_RUN_LOG = 'run.log'
DEFAULT_RESULTS_TSV = 'results.tsv'
DEFAULT_EXPERIMENT_TIMEOUT_SECONDS = 600
TRAIN_COMMAND = 'uv run train.py > run.log 2>&1'


class AutoresearchError(RuntimeError):
    """Raised when autoresearch setup or experiment execution fails."""


@dataclass(frozen=True)
class AutoresearchSetupReport:
    root: str
    branch: str
    data_ready: bool
    tokenizer_ready: bool
    results_tsv_path: str
    results_tsv_initialized: bool
    cache_dir: str
    suggested_tag: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentMetrics:
    success: bool
    log_path: str
    timed_out: bool
    return_code: int
    val_bpb: float | None = None
    peak_vram_mb: float | None = None
    training_seconds: float | None = None
    total_seconds: float | None = None
    mfu_percent: float | None = None
    total_tokens_m: float | None = None
    num_steps: int | None = None
    num_params_m: float | None = None
    depth: int | None = None
    error: str | None = None
    tail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_autoresearch_root(root: str | Path | None = None) -> Path:
    if root is not None:
        candidate = Path(root).resolve()
        _validate_root(candidate)
        return candidate

    cwd = Path.cwd().resolve()
    if _looks_like_autoresearch_root(cwd):
        return cwd

    workspace_candidate = Path(__file__).resolve().parents[3] / 'nodes' / 'autoresearch-macos'
    if workspace_candidate.exists() and _looks_like_autoresearch_root(workspace_candidate):
        return workspace_candidate

    raise AutoresearchError(
        'Could not resolve autoresearch root automatically. Pass --root pointing at a repo with '
        'prepare.py, train.py, and program.md.'
    )


def setup_autoresearch(root: str | Path | None = None, initialize_results: bool = True) -> AutoresearchSetupReport:
    repo_root = resolve_autoresearch_root(root)
    branch = _git_output(repo_root, ['git', 'branch', '--show-current']) or 'DETACHED'
    cache_dir = Path.home() / '.cache' / 'autoresearch'
    data_ready = (cache_dir / 'data').exists() and any((cache_dir / 'data').glob('shard_*.parquet'))
    tokenizer_ready = (cache_dir / 'tokenizer' / 'tokenizer.pkl').exists() and (cache_dir / 'tokenizer' / 'token_bytes.pt').exists()
    results_path = repo_root / DEFAULT_RESULTS_TSV
    initialized = False
    if initialize_results and not results_path.exists():
        results_path.write_text(RESULTS_HEADER, encoding='utf-8')
        initialized = True
    suggested_tag = datetime.now().strftime('%b%d').lower()
    return AutoresearchSetupReport(
        root=str(repo_root),
        branch=branch,
        data_ready=data_ready,
        tokenizer_ready=tokenizer_ready,
        results_tsv_path=str(results_path),
        results_tsv_initialized=initialized,
        cache_dir=str(cache_dir),
        suggested_tag=suggested_tag,
    )


def parse_run_log(path: str | Path) -> ExperimentMetrics:
    log_path = Path(path)
    if not log_path.exists():
        raise AutoresearchError(f'Run log not found: {log_path}')
    text = log_path.read_text(encoding='utf-8', errors='replace')
    metrics = _extract_metrics(text)
    if metrics.get('val_bpb') is None:
        tail = '\n'.join(text.splitlines()[-50:])
        return ExperimentMetrics(
            success=False,
            log_path=str(log_path),
            timed_out=False,
            return_code=1,
            error='run log did not contain a final val_bpb summary',
            tail=tail,
        )
    return ExperimentMetrics(
        success=True,
        log_path=str(log_path),
        timed_out=False,
        return_code=0,
        val_bpb=metrics.get('val_bpb'),
        peak_vram_mb=metrics.get('peak_vram_mb'),
        training_seconds=metrics.get('training_seconds'),
        total_seconds=metrics.get('total_seconds'),
        mfu_percent=metrics.get('mfu_percent'),
        total_tokens_m=metrics.get('total_tokens_M'),
        num_steps=int(metrics['num_steps']) if metrics.get('num_steps') is not None else None,
        num_params_m=metrics.get('num_params_M'),
        depth=int(metrics['depth']) if metrics.get('depth') is not None else None,
    )


def run_experiment(
    root: str | Path | None = None,
    command: str = TRAIN_COMMAND,
    log_path: str | Path = DEFAULT_RUN_LOG,
    timeout_seconds: int = DEFAULT_EXPERIMENT_TIMEOUT_SECONDS,
) -> ExperimentMetrics:
    repo_root = resolve_autoresearch_root(root)
    target_log = Path(log_path)
    if not target_log.is_absolute():
        target_log = repo_root / target_log

    try:
        run = subprocess.run(
            ['/bin/zsh', '-lc', command],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        tail = ''
        if target_log.exists():
            tail = '\n'.join(target_log.read_text(encoding='utf-8', errors='replace').splitlines()[-50:])
        return ExperimentMetrics(
            success=False,
            log_path=str(target_log),
            timed_out=True,
            return_code=-1,
            error=f'experiment command timed out after {timeout_seconds} seconds',
            tail=tail,
        )
    if run.returncode != 0:
        tail = ''
        if target_log.exists():
            tail = '\n'.join(target_log.read_text(encoding='utf-8', errors='replace').splitlines()[-50:])
        else:
            tail = '\n'.join(part for part in (run.stdout.strip(), run.stderr.strip()) if part)
        return ExperimentMetrics(
            success=False,
            log_path=str(target_log),
            timed_out=False,
            return_code=run.returncode,
            error='experiment command failed',
            tail=tail,
        )

    metrics = parse_run_log(target_log)
    return ExperimentMetrics(
        **{
            **metrics.to_dict(),
            'success': metrics.success,
            'return_code': run.returncode,
            'timed_out': False,
        }
    )


def append_results_row(
    root: str | Path | None,
    commit: str,
    metrics: ExperimentMetrics,
    status: str,
    description: str,
    results_path: str | Path = DEFAULT_RESULTS_TSV,
) -> Path:
    repo_root = resolve_autoresearch_root(root)
    target = Path(results_path)
    if not target.is_absolute():
        target = repo_root / target
    if not target.exists():
        target.write_text(RESULTS_HEADER, encoding='utf-8')
    val_bpb = metrics.val_bpb if metrics.val_bpb is not None else 0.0
    memory_gb = (metrics.peak_vram_mb or 0.0) / 1024
    row = f'{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n'
    with target.open('a', encoding='utf-8') as handle:
        handle.write(row)
    return target


def best_recorded_bpb(root: str | Path | None, results_path: str | Path = DEFAULT_RESULTS_TSV) -> float | None:
    repo_root = resolve_autoresearch_root(root)
    target = Path(results_path)
    if not target.is_absolute():
        target = repo_root / target
    if not target.exists():
        return None
    best: float | None = None
    with target.open('r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        for row in reader:
            if row.get('status') == 'crash':
                continue
            try:
                value = float(row['val_bpb'])
            except (TypeError, ValueError, KeyError):
                continue
            if best is None or value < best:
                best = value
    return best


def short_head_commit(root: str | Path | None) -> str:
    repo_root = resolve_autoresearch_root(root)
    commit = _git_output(repo_root, ['git', 'rev-parse', '--short', 'HEAD'])
    return commit or 'unknown'


def _validate_root(root: Path) -> None:
    if not _looks_like_autoresearch_root(root):
        raise AutoresearchError(
            f'{root} does not look like an autoresearch repo. Expected prepare.py, train.py, and program.md.'
        )


def _looks_like_autoresearch_root(path: Path) -> bool:
    return (path / 'prepare.py').exists() and (path / 'train.py').exists() and (path / 'program.md').exists()


def _git_output(root: Path, args: list[str]) -> str:
    result = subprocess.run(args, cwd=root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return ''
    return result.stdout.strip()


def _extract_metrics(text: str) -> dict[str, float]:
    keys = (
        'val_bpb',
        'training_seconds',
        'total_seconds',
        'peak_vram_mb',
        'mfu_percent',
        'total_tokens_M',
        'num_steps',
        'num_params_M',
        'depth',
    )
    metrics: dict[str, float] = {}
    for key in keys:
        match = re.search(rf'^{re.escape(key)}:\s+([0-9.]+)$', text, flags=re.MULTILINE)
        if match:
            metrics[key] = float(match.group(1))
    return metrics
