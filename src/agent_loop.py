from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm_backend import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL, LLMBackendError, OllamaBackend
from .permissions import ToolPermissionContext
from .runtime_tools import RuntimeToolRegistry, render_tool_instructions

DEFAULT_AGENT_SYSTEM_PROMPT = (
    'You are a local coding assistant running inside a minimal harness. '
    'Use tools when needed. Respond with exactly one JSON object and no surrounding prose. '
    'Valid outputs are {"type":"tool_call","name":"...","arguments":{...}} or '
    '{"type":"final","content":"..."}. '
    'Do not wrap JSON in markdown fences.'
)


@dataclass(frozen=True)
class AgentRunResult:
    content: str
    turns: int
    tool_calls: tuple[str, ...]
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
) -> AgentRunResult:
    workspace_root = (root or Path.cwd()).resolve()
    backend = OllamaBackend(model=model, host=host)
    tools = RuntimeToolRegistry(root=workspace_root, permission_context=permission_context)
    messages = build_agent_messages(prompt=prompt, workspace_root=workspace_root, system_prompt=system_prompt)
    tool_calls: list[str] = []
    trace_events: list[str] = []
    tool_required = prompt_requires_tool(prompt)
    fallback_used = False

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
                _append_tool_result(messages, tools, fallback, tool_calls, trace_events if trace else None, turn)
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
                    _append_tool_result(messages, tools, fallback, tool_calls, trace_events if trace else None, turn)
                    continue
                messages.append({'role': 'user', 'content': tool_required_message()})
                continue
            return AgentRunResult(
                content=payload['content'],
                turns=turn,
                tool_calls=tuple(tool_calls),
                trace_events=tuple(trace_events),
            )

        _append_tool_result(messages, tools, payload, tool_calls, trace_events if trace else None, turn)

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


def prompt_requires_tool(prompt: str) -> bool:
    lowered = prompt.lower()
    action_words = ('read', 'search', 'find', 'inspect', 'open', 'edit', 'modify', 'create', 'write', 'run', 'execute', 'quote')
    target_hints = ('src/', '.py', 'file', 'files', 'workspace', 'tree', 'directory', 'lines', 'command', 'shell', 'pwd', 'ls ', 'rg ', 'grep ')
    return any(word in lowered for word in action_words) and any(hint in lowered for hint in target_hints)


def infer_tool_call_from_prompt(prompt: str) -> dict[str, Any] | None:
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

    return None


def _append_tool_result(
    messages: list[dict[str, str]],
    tools: RuntimeToolRegistry,
    payload: dict[str, Any],
    tool_calls: list[str],
    trace_events: list[str] | None,
    turn: int,
) -> None:
    tool_name = payload['name']
    arguments = payload['arguments']
    tool_calls.append(tool_name)
    if trace_events is not None:
        trace_events.append(f'tool_call[{turn}]={tool_name} {json.dumps(arguments, ensure_ascii=True)}')
    tool_result = tools.execute(tool_name, arguments)
    if trace_events is not None:
        trace_events.append(
            f'tool_result[{turn}]={tool_name} success={tool_result.success} '
            f'error={tool_result.error!r}'
        )
    messages.append({'role': 'user', 'content': tool_result_message(tool_result.to_payload())})
