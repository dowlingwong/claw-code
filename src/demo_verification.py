from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_PATHS = ('train.py', 'prepare.py', 'program.md', 'README.md')


def snapshot_workspace_sources(root: str | Path, paths: tuple[str, ...] = DEFAULT_SOURCE_PATHS) -> dict[str, str]:
    repo_root = Path(root).resolve()
    snapshot: dict[str, str] = {}
    for relative_path in paths:
        target = repo_root / relative_path
        if target.exists() and target.is_file():
            snapshot[relative_path] = target.read_text(encoding='utf-8', errors='replace')
    return snapshot


def get_changed_files(before_snapshot: dict[str, str], after_snapshot: dict[str, str]) -> list[str]:
    changed = {
        path
        for path in set(before_snapshot) | set(after_snapshot)
        if before_snapshot.get(path) != after_snapshot.get(path)
    }
    return sorted(changed)


def build_diff_summary(before_snapshot: dict[str, str], after_snapshot: dict[str, str]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for path in get_changed_files(before_snapshot, after_snapshot):
        before_text = before_snapshot.get(path, '')
        after_text = after_snapshot.get(path, '')
        diff_lines = list(
            difflib.unified_diff(
                before_text.splitlines(),
                after_text.splitlines(),
                fromfile=f'a/{path}',
                tofile=f'b/{path}',
                lineterm='',
            )
        )
        added = sum(1 for line in diff_lines if line.startswith('+') and not line.startswith('+++'))
        removed = sum(1 for line in diff_lines if line.startswith('-') and not line.startswith('---'))
        summary.append(
            {
                'path': path,
                'added_lines': added,
                'removed_lines': removed,
                'changed_line_count': added + removed,
                'diff': '\n'.join(diff_lines),
            }
        )
    return summary


def extract_metric_from_text(text: str, key: str = 'val_bpb') -> float | None:
    pattern = re.compile(
        rf'^\s*{re.escape(key)}\s*[:=]\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$',
        flags=re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return None
    return float(match.group(1))


def tail_lines(text: str, limit: int = 12) -> str:
    lines = text.splitlines()
    return '\n'.join(lines[-limit:])
