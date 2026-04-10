from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskPacket:
    objective: str
    scope: str
    repo: str
    branch_policy: str
    acceptance_tests: tuple[str, ...]
    commit_policy: str
    reporting_contract: str
    escalation_policy: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['acceptance_tests'] = list(self.acceptance_tests)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'TaskPacket':
        acceptance_tests = payload.get('acceptance_tests', ())
        if not isinstance(acceptance_tests, list):
            raise TaskPacketValidationError(['acceptance_tests must be a list'])
        packet = cls(
            objective=str(payload.get('objective', '')),
            scope=str(payload.get('scope', '')),
            repo=str(payload.get('repo', '')),
            branch_policy=str(payload.get('branch_policy', '')),
            acceptance_tests=tuple(str(test) for test in acceptance_tests),
            commit_policy=str(payload.get('commit_policy', '')),
            reporting_contract=str(payload.get('reporting_contract', '')),
            escalation_policy=str(payload.get('escalation_policy', '')),
        )
        validate_task_packet(packet)
        return packet


class TaskPacketValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__('; '.join(errors))


def validate_task_packet(packet: TaskPacket) -> None:
    errors: list[str] = []
    for field_name in (
        'objective',
        'scope',
        'repo',
        'branch_policy',
        'commit_policy',
        'reporting_contract',
        'escalation_policy',
    ):
        if not getattr(packet, field_name).strip():
            errors.append(f'{field_name} must not be empty')
    for index, test in enumerate(packet.acceptance_tests):
        if not test.strip():
            errors.append(f'acceptance_tests contains an empty value at index {index}')
    if errors:
        raise TaskPacketValidationError(errors)


def load_task_packet(path: str | Path) -> TaskPacket:
    packet_path = Path(path)
    payload = json.loads(packet_path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise TaskPacketValidationError(['task packet must be a JSON object'])
    return TaskPacket.from_dict(payload)


def render_worker_prompt(packet: TaskPacket) -> str:
    lines = [
        'You are operating as a local worker node in a managed coding harness.',
        f'Objective: {packet.objective}',
        f'Scope: {packet.scope}',
        f'Repository: {packet.repo}',
        f'Branch policy: {packet.branch_policy}',
        f'Commit policy: {packet.commit_policy}',
        f'Reporting contract: {packet.reporting_contract}',
        f'Escalation policy: {packet.escalation_policy}',
    ]
    if packet.acceptance_tests:
        lines.append('Acceptance tests the manager will run after you finish:')
        lines.extend(f'- {command}' for command in packet.acceptance_tests)
    if _is_readonly_packet(packet):
        lines.append('Use tools to inspect the workspace and return a concise final answer.')
    else:
        lines.append('Make the required code or documentation changes in the workspace and return a concise final answer describing what changed and any blockers.')
    return '\n'.join(lines)


def _is_readonly_packet(packet: TaskPacket) -> bool:
    """Return True when the packet's scope explicitly prohibits file edits.

    A task is considered read-only when its scope starts with (or is entirely
    composed of) read-only language AND does not open with a mutation verb.
    Tasks like the autoresearch packet that say "Modify only train.py … Do not
    edit prepare.py" are edit tasks — the leading verb wins.
    """
    scope = packet.scope.lower().strip()
    # If the scope opens with a mutation verb the task is definitely an edit task.
    _EDIT_OPENERS = ('modify ', 'edit ', 'change ', 'update ', 'write ', 'create ', 'add ')
    if any(scope.startswith(v) for v in _EDIT_OPENERS):
        return False
    # Otherwise look for readonly signals at the start of the scope.
    _READONLY_SIGNALS = ('do not edit', 'not edit', 'no edit', 'read only', 'read-only', 'read ')
    return any(scope.startswith(sig) or (f' {sig}' in scope[:60]) for sig in _READONLY_SIGNALS)
