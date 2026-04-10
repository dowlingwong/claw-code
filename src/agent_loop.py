from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm_backend import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL, LLMBackend, LLMBackendError, OllamaBackend, create_backend
from .permissions import ToolPermissionContext
from .runtime_tools import RuntimeToolRegistry, render_tool_instructions

DEFAULT_AGENT_SYSTEM_PROMPT = (
    'You are a local coding assistant running inside a minimal harness. '
    'Use tools when needed. Respond with exactly one JSON object and no surrounding prose. '
    'Valid outputs are {"type":"tool_call","name":"...","arguments":{...}} or '
    '{"type":"final","content":"..."}. '
    'Do not wrap JSON in markdown fences.'
)
MUTATING_TOOL_NAMES = frozenset({'edit_file', 'run_shell_command'})


@dataclass(frozen=True)
class AgentRunResult:
    content: str
    turns: int
    tool_calls: tuple[str, ...]
    tool_trace: tuple[dict[str, Any], ...] = ()
    stop_reason: str = 'completed'
    trace_events: tuple[str, ...] = ()


class AgentProtocolError(RuntimeError):
    """Raised when the model does not follow the JSON tool-call protocol."""


def run_agent_task(
    prompt: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    root: Path | None = None,
    system_prompt: str = DEFAULT_AGENT_SYSTEM_PROMPT,
    max_turns: int = 8,
    permission_context: ToolPermissionContext | None = None,
    trace: bool = False,
    backend: LLMBackend | None = None,
) -> AgentRunResult:
    workspace_root = (root or Path.cwd()).resolve()
    if backend is None:
        backend = OllamaBackend(model=model, host=host)
    tools = RuntimeToolRegistry(root=workspace_root, permission_context=permission_context)
    messages = build_agent_messages(prompt=prompt, workspace_root=workspace_root, system_prompt=system_prompt)
    tool_calls: list[str] = []
    tool_trace: list[dict[str, Any]] = []
    trace_events: list[str] = []
    tool_required = prompt_requires_tool(prompt)
    edit_required = prompt_requires_edit(prompt)
    required_edit_target = infer_required_edit_target(prompt)
    fallback_used = False
    # Tracks whether a successful file edit has been confirmed in the tool trace.
    # Once True, the model is allowed to return a final answer without being
    # asked to edit again — this enables early exit as soon as the edit lands
    # rather than always running up to max_turns.
    edit_confirmed = False

    for turn in range(1, max_turns + 1):
        response = backend.chat(messages, response_format='json', options={'temperature': 0})
        assistant_content = response.content.strip()
        if trace:
            trace_events.append(f'assistant_raw[{turn}]={assistant_content}')
        messages.append({'role': 'assistant', 'content': assistant_content})

        try:
            payload = parse_agent_response(assistant_content)
        except AgentProtocolError as error:
            if trace:
                trace_events.append(f'protocol_error[{turn}]={error}')
            fallback = None
            if tool_required and not tool_calls and not fallback_used:
                fallback = infer_tool_call_from_prompt(prompt)
            if fallback is not None:
                fallback_used = True
                if trace:
                    trace_events.append(f'fallback_tool[{turn}]={fallback["name"]} {json.dumps(fallback["arguments"], ensure_ascii=True)}')
                _append_tool_result(messages, tools, fallback, tool_calls, tool_trace, trace_events if trace else None, turn)
                continue
            messages.append({'role': 'user', 'content': protocol_error_message(str(error))})
            continue

        if payload['type'] == 'final':
            if tool_required and not tool_calls:
                if trace:
                    trace_events.append(f'tool_required_reject[{turn}]={payload["content"]}')
                fallback = None
                if not fallback_used:
                    fallback = infer_tool_call_from_prompt(prompt)
                if fallback is not None:
                    fallback_used = True
                    if trace:
                        trace_events.append(f'fallback_tool[{turn}]={fallback["name"]} {json.dumps(fallback["arguments"], ensure_ascii=True)}')
                    _append_tool_result(messages, tools, fallback, tool_calls, tool_trace, trace_events if trace else None, turn)
                    continue
                messages.append({'role': 'user', 'content': tool_required_message()})
                continue
            if edit_required and not edit_confirmed and not has_successful_edit(tool_trace, required_edit_target):
                if trace:
                    trace_events.append(f'edit_required_reject[{turn}]={payload["content"]}')
                fallback = None
                if not fallback_used:
                    fallback = infer_tool_call_from_prompt(prompt, prefer_edit=True)
                if fallback is not None:
                    fallback_used = True
                    if trace:
                        trace_events.append(f'fallback_tool[{turn}]={fallback["name"]} {json.dumps(fallback["arguments"], ensure_ascii=True)}')
                    _append_tool_result(messages, tools, fallback, tool_calls, tool_trace, trace_events if trace else None, turn)
                    continue
                messages.append({'role': 'user', 'content': edit_required_message()})
                continue
            return AgentRunResult(
                content=payload['content'],
                turns=turn,
                tool_calls=tuple(tool_calls),
                tool_trace=tuple(tool_trace),
                stop_reason='completed',
                trace_events=tuple(trace_events),
            )

        _append_tool_result(messages, tools, payload, tool_calls, tool_trace, trace_events if trace else None, turn)
        # Check whether the edit was confirmed by this tool call so the model
        # can exit on the very next turn rather than being forced to keep going.
        if edit_required and not edit_confirmed:
            edit_confirmed = has_successful_edit(tool_trace, required_edit_target)
            if edit_confirmed and trace:
                trace_events.append(f'edit_confirmed[{turn}]={required_edit_target!r}')

    if trace:
        trace_events.append(f'stopped=max_turns:{max_turns}')
    raise LLMBackendError(f'Agent stopped after reaching max_turns={max_turns}.')


def build_agent_messages(
    prompt: str,
    workspace_root: Path,
    system_prompt: str = DEFAULT_AGENT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    return [
        {
            'role': 'system',
            'content': '\n\n'.join(
                [
                    system_prompt,
                    f'Workspace root: {workspace_root}',
                    render_tool_instructions(),
                    'After every tool call you will receive a user message starting with TOOL_RESULT followed by JSON.',
                    'If the user asks to inspect workspace files, search code, edit files, or run shell commands, use a tool before giving a final answer.',
                    'Example tool call: {"type":"tool_call","name":"read_file","arguments":{"path":"src/main.py","start_line":1,"end_line":40}}',
                    'Example final: {"type":"final","content":"I found ..."}',
                ]
            ),
        },
        {'role': 'user', 'content': prompt},
    ]


def parse_agent_response(content: str) -> dict[str, Any]:
    normalized = content.strip()
    if normalized.startswith('```'):
        lines = normalized.splitlines()
        if lines and lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].startswith('```'):
            lines = lines[:-1]
        normalized = '\n'.join(lines).strip()

    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError as error:
        raise AgentProtocolError('Response was not valid JSON.') from error

    if not isinstance(payload, dict):
        raise AgentProtocolError('Response JSON must be an object.')

    response_type = payload.get('type')
    if response_type == 'final':
        content_value = payload.get('content')
        if not isinstance(content_value, str) or not content_value:
            raise AgentProtocolError('Final responses must include a non-empty string content field.')
        return {'type': 'final', 'content': content_value}

    if response_type == 'tool_call':
        name = payload.get('name')
        arguments = payload.get('arguments')
        if not isinstance(name, str) or not name:
            raise AgentProtocolError('Tool calls must include a non-empty string name field.')
        if not isinstance(arguments, dict):
            raise AgentProtocolError('Tool calls must include an arguments object.')
        return {'type': 'tool_call', 'name': name, 'arguments': arguments}

    raise AgentProtocolError('Response type must be either "tool_call" or "final".')


def tool_result_message(payload: dict[str, Any]) -> str:
    return f'TOOL_RESULT {json.dumps(payload, ensure_ascii=True)}'


def protocol_error_message(error: str) -> str:
    return (
        'Your previous response did not follow the protocol. '
        f'{error} '
        'Respond again with exactly one JSON object.'
    )


def tool_required_message() -> str:
    return (
        'You must use an appropriate tool before answering this request because it depends on workspace contents '
        'or shell output. Respond with exactly one JSON tool_call object.'
    )


def edit_required_message() -> str:
    return (
        'You must make a real workspace change before answering this request. '
        'A read-only inspection is insufficient. Respond with exactly one JSON tool_call object that performs the edit.'
    )


def prompt_requires_tool(prompt: str) -> bool:
    lowered = prompt.lower()
    action_words = ('read', 'search', 'find', 'inspect', 'open', 'edit', 'modify', 'create', 'write', 'run', 'execute', 'quote')
    target_hints = ('src/', '.py', 'file', 'files', 'workspace', 'tree', 'directory', 'lines', 'command', 'shell', 'pwd', 'ls ', 'rg ', 'grep ')
    return any(word in lowered for word in action_words) and any(hint in lowered for hint in target_hints)


def prompt_requires_edit(prompt: str) -> bool:
    """Return True only when the prompt contains an un-negated mutating verb near a file target.

    Negation prefixes like "do not edit", "don't modify", "never write" within a
    10-character window before the verb are treated as read-only instructions so
    that scope text such as "Do not edit any file" does not falsely set the flag.
    """
    lowered = prompt.lower()
    edit_words = ('edit', 'modify', 'change', 'update', 'write', 'create', 'add')
    negations = ('not ', "don't ", 'never ', 'no ')
    target_hints = ('.py', '.md', '.txt', 'file', 'workspace', 'train.py', 'src/')

    has_affirmative_edit = False
    for word in edit_words:
        pos = 0
        while True:
            idx = lowered.find(word, pos)
            if idx == -1:
                break
            # Look at up to 12 characters before the verb for negation markers
            prefix = lowered[max(0, idx - 12):idx]
            if not any(neg in prefix for neg in negations):
                has_affirmative_edit = True
                break
            pos = idx + 1
        if has_affirmative_edit:
            break

    if not has_affirmative_edit:
        return False
    return any(hint in lowered for hint in target_hints)


def infer_required_edit_target(prompt: str) -> str | None:
    """Extract the file path that should be edited from the prompt.

    Returns None when no concrete path-like token is found (e.g. the matched
    word is a common English word like 'any', 'file', 'it', or 'this').
    """
    _GENERIC_WORDS = frozenset({
        'any', 'file', 'files', 'it', 'this', 'that', 'them', 'here', 'there',
        'code', 'module', 'script', 'document', 'content', 'text',
    })
    match = re.search(
        r'(?:modify|edit|change|update|write|create|add(?:\s+to)?)\s+(?P<path>[\w./-]+)',
        prompt,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    path = match.group('path')
    # Reject generic words that are not real file paths
    if path.lower() in _GENERIC_WORDS or '.' not in path:
        return None
    return path


def infer_tool_call_from_prompt(prompt: str, prefer_edit: bool = False) -> dict[str, Any] | None:
    if prefer_edit:
        edit_fallback = infer_edit_tool_call_from_prompt(prompt)
        if edit_fallback is not None:
            return edit_fallback

    edit_fallback = infer_edit_tool_call_from_prompt(prompt)
    if edit_fallback is not None:
        return edit_fallback

    read_match = re.search(
        r'read\s+(?P<path>[\w./-]+)(?:\s+lines?\s+(?P<start>\d+)\s*(?:to|-)\s*(?P<end>\d+))?',
        prompt,
        flags=re.IGNORECASE,
    )
    if read_match:
        arguments: dict[str, Any] = {'path': read_match.group('path')}
        if read_match.group('start') and read_match.group('end'):
            arguments['start_line'] = int(read_match.group('start'))
            arguments['end_line'] = int(read_match.group('end'))
        return {'name': 'read_file', 'arguments': arguments}

    search_match = re.search(
        r"search(?:\s+the)?\s+(?P<scope>.+?)\s+for\s+['\"](?P<pattern>[^'\"]+)['\"]",
        prompt,
        flags=re.IGNORECASE,
    )
    if search_match:
        scope = search_match.group('scope').strip().lower()
        path = '.'
        if 'src' in scope:
            path = 'src'
        return {
            'name': 'search_files',
            'arguments': {'path': path, 'pattern': search_match.group('pattern')},
        }

    quoted_commands = re.findall(r"['\"]([^'\"]+)['\"]", prompt)
    if any(word in prompt.lower() for word in ('run', 'execute', 'shell')) and quoted_commands:
        return {
            'name': 'run_shell_command',
            'arguments': {'command': '\n'.join(quoted_commands)},
        }

    create_match = re.search(
        r'create\s+(?P<path>[\w./-]+)\s+containing\s+(?P<content>.+)',
        prompt,
        flags=re.IGNORECASE,
    )
    if create_match:
        return {
            'name': 'edit_file',
            'arguments': {
                'path': create_match.group('path'),
                'new_text': create_match.group('content').strip(),
            },
        }

    if 'git status' in prompt.lower():
        return {'name': 'git_status', 'arguments': {}}

    if 'git diff' in prompt.lower():
        return {'name': 'git_diff', 'arguments': {}}

    if 'run tests' in prompt.lower() or 'test suite' in prompt.lower():
        return {'name': 'run_tests', 'arguments': {}}

    if 'run build' in prompt.lower() or 'build the project' in prompt.lower():
        return {'name': 'run_build', 'arguments': {}}

    if 'list directory' in prompt.lower() or 'list dir' in prompt.lower():
        return {'name': 'list_dir', 'arguments': {'path': '.'}}

    return None


def infer_edit_tool_call_from_prompt(prompt: str) -> dict[str, Any] | None:
    modify_match = re.search(
        r'(?:modify|edit|change|update)\s+(?P<path>[\w./-]+)\s+by\s+(?P<details>.+?)(?:(?<!\d)\.(?!\d)|\n|$)',
        prompt,
        flags=re.IGNORECASE,
    )
    if modify_match:
        path = modify_match.group('path')
        details = modify_match.group('details').strip().lower()
        assignment_match = re.search(
            r'(?:set|setting)\s+`?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>[-+0-9.eE]+|True|False|None|".*?"|\'.*?\')`?',
            modify_match.group('details').strip(),
            flags=re.IGNORECASE,
        )
        if assignment_match:
            variable_name = assignment_match.group('name')
            replacement_value = assignment_match.group('value')
            command = '\n'.join(
                [
                    "python3 - <<'PY'",
                    'from pathlib import Path',
                    'import json',
                    'import re',
                    f"target = Path({json.dumps(path)})",
                    "text = target.read_text(encoding='utf-8')",
                    f"name = {json.dumps(variable_name)}",
                    f"value = {json.dumps(replacement_value)}",
                    "pattern = re.compile(rf'^(?P<indent>\\s*){re.escape(name)}\\s*=.*$', flags=re.MULTILINE)",
                    "replacement = f'{name} = {value}'",
                    "updated, count = pattern.subn(replacement, text, count=1)",
                    'if count != 1:',
                    "    raise SystemExit(f'assignment not found: {name}')",
                    "target.write_text(updated, encoding='utf-8')",
                    'PY',
                ]
            )
            return {
                'name': 'run_shell_command',
                'arguments': {'command': command, 'timeout_seconds': 20},
            }
        if 'comment' in details:
            marker = '# smoke-test comment: '
            command = '\n'.join(
                [
                    "python3 - <<'PY'",
                    'from pathlib import Path',
                    'import re',
                    f"target = Path({json.dumps(path)})",
                    "text = target.read_text(encoding='utf-8')",
                    f"marker = {json.dumps(marker)}",
                    "match = re.match(r'^# smoke-test comment: (\\d+)\\n', text)",
                    'if match:',
                    '    current = int(match.group(1))',
                    "    updated = f'{marker}{current + 1}\\n' + text[match.end():]",
                    'else:',
                    "    updated = f'{marker}1\\n' + text",
                    "target.write_text(updated, encoding='utf-8')",
                    'PY',
                ]
            )
            return {
                'name': 'run_shell_command',
                'arguments': {'command': command, 'timeout_seconds': 20},
            }
    return None


def has_successful_edit(tool_trace: list[dict[str, Any]], required_path: str | None) -> bool:
    for entry in tool_trace:
        if not entry.get('success'):
            continue
        if entry.get('name') not in MUTATING_TOOL_NAMES:
            continue
        metadata = entry.get('metadata')
        if required_path is None:
            return True
        if isinstance(metadata, dict) and metadata.get('path') == required_path:
            return True
        arguments = entry.get('arguments')
        if isinstance(arguments, dict) and arguments.get('path') == required_path:
            return True
        if (
            entry.get('name') == 'run_shell_command'
            and isinstance(arguments, dict)
            and isinstance(arguments.get('command'), str)
            and required_path in arguments['command']
        ):
            return True
    return False


def _append_tool_result(
    messages: list[dict[str, str]],
    tools: RuntimeToolRegistry,
    payload: dict[str, Any],
    tool_calls: list[str],
    tool_trace: list[dict[str, Any]],
    trace_events: list[str] | None,
    turn: int,
) -> None:
    tool_name = payload['name']
    arguments = payload['arguments']
    tool_calls.append(tool_name)
    if trace_events is not None:
        trace_events.append(f'tool_call[{turn}]={tool_name} {json.dumps(arguments, ensure_ascii=True)}')
    tool_result = tools.execute(tool_name, arguments)
    tool_trace.append(
        {
            'turn': turn,
            'name': tool_name,
            'arguments': arguments,
            'success': tool_result.success,
            'output': tool_result.output,
            'metadata': tool_result.metadata,
            'error': tool_result.error,
        }
    )
    if trace_events is not None:
        trace_events.append(
            f'tool_result[{turn}]={tool_name} success={tool_result.success} '
            f'error={tool_result.error!r}'
        )
    messages.append({'role': 'user', 'content': tool_result_message(tool_result.to_payload())})
