from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from shutil import which
from typing import Any

from .permissions import ToolPermissionContext

MAX_TOOL_OUTPUT_CHARS = 12_000


class RuntimeToolError(RuntimeError):
    """Raised when a runtime tool cannot complete a request."""


@dataclass(frozen=True)
class RuntimeToolDefinition:
    name: str
    description: str
    arguments: dict[str, str]


@dataclass(frozen=True)
class RuntimeToolResult:
    name: str
    success: bool
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            'name': self.name,
            'success': self.success,
            'output': self.output,
            'metadata': self.metadata,
        }
        if self.error is not None:
            payload['error'] = self.error
        return payload


TOOL_DEFINITIONS = (
    RuntimeToolDefinition(
        name='list_dir',
        description='List directory entries inside the workspace root.',
        arguments={
            'path': 'Optional workspace-relative directory path. Defaults to the workspace root.',
            'max_entries': 'Optional cap on returned entries. Defaults to 200.',
        },
    ),
    RuntimeToolDefinition(
        name='read_file',
        description='Read a file from the workspace. Optionally limit the response to a line range.',
        arguments={
            'path': 'Workspace-relative file path.',
            'start_line': 'Optional 1-based starting line number.',
            'end_line': 'Optional 1-based ending line number.',
        },
    ),
    RuntimeToolDefinition(
        name='search_files',
        description='Search workspace files for a regular expression pattern and return matching lines.',
        arguments={
            'pattern': 'Regular expression to search for.',
            'path': 'Optional workspace-relative directory to search from. Defaults to the workspace root.',
            'max_results': 'Optional cap on returned matches. Defaults to 50.',
        },
    ),
    RuntimeToolDefinition(
        name='edit_file',
        description='Edit a workspace file by replacing an exact old_text snippet with new_text. Creates a new file when old_text is omitted and the file does not exist.',
        arguments={
            'path': 'Workspace-relative file path.',
            'old_text': 'Exact existing text to replace. Required when editing an existing file.',
            'new_text': 'Replacement text to write.',
        },
    ),
    RuntimeToolDefinition(
        name='run_shell_command',
        description='Run a shell command inside the workspace and capture stdout, stderr, and exit code.',
        arguments={
            'command': 'Shell command string to execute.',
            'timeout_seconds': 'Optional timeout in seconds. Defaults to 20.',
        },
    ),
    RuntimeToolDefinition(
        name='git_status',
        description='Show git status for the current workspace.',
        arguments={},
    ),
    RuntimeToolDefinition(
        name='git_diff',
        description='Show git diff output, optionally scoped to a path or staged changes.',
        arguments={
            'path': 'Optional workspace-relative path to diff.',
            'staged': 'Optional boolean. When true, show staged diff.',
        },
    ),
    RuntimeToolDefinition(
        name='run_tests',
        description='Run the project test command and capture the result.',
        arguments={
            'command': 'Optional shell command. Defaults to python3 -m unittest discover -s tests -v.',
            'timeout_seconds': 'Optional timeout in seconds. Defaults to 120.',
        },
    ),
    RuntimeToolDefinition(
        name='run_build',
        description='Run the project build command and capture the result.',
        arguments={
            'command': 'Optional shell command. Defaults to python3 -m compileall src.',
            'timeout_seconds': 'Optional timeout in seconds. Defaults to 120.',
        },
    ),
)


def render_tool_instructions() -> str:
    lines = [
        'Available tools:',
    ]
    for tool in TOOL_DEFINITIONS:
        lines.append(f'- {tool.name}: {tool.description}')
        for arg, detail in tool.arguments.items():
            lines.append(f'  - {arg}: {detail}')
    return '\n'.join(lines)


@dataclass
class RuntimeToolRegistry:
    root: Path
    permission_context: ToolPermissionContext | None = None
    shell: str = '/bin/zsh'

    def __post_init__(self) -> None:
        self.root = self.root.resolve()

    def execute(self, name: str, arguments: dict[str, Any]) -> RuntimeToolResult:
        if self.permission_context and self.permission_context.blocks(name):
            return RuntimeToolResult(
                name=name,
                success=False,
                output='',
                error=f'Tool is blocked by permission policy: {name}',
            )

        handlers = {
            'list_dir': self._list_dir,
            'read_file': self._read_file,
            'search_files': self._search_files,
            'edit_file': self._edit_file,
            'run_shell_command': self._run_shell_command,
            'git_status': self._git_status,
            'git_diff': self._git_diff,
            'run_tests': self._run_tests,
            'run_build': self._run_build,
        }
        handler = handlers.get(name)
        if handler is None:
            return RuntimeToolResult(
                name=name,
                success=False,
                output='',
                error=f'Unknown runtime tool: {name}',
            )

        try:
            return handler(arguments)
        except RuntimeToolError as error:
            return RuntimeToolResult(name=name, success=False, output='', error=str(error))

    def _read_file(self, arguments: dict[str, Any]) -> RuntimeToolResult:
        path = self._resolve_path(self._require_string(arguments, 'path'))
        if not path.exists():
            raise RuntimeToolError(f'File not found: {path.relative_to(self.root)}')
        if not path.is_file():
            raise RuntimeToolError(f'Path is not a file: {path.relative_to(self.root)}')

        text = path.read_text(encoding='utf-8')
        lines = text.splitlines()
        start_line = self._optional_int(arguments, 'start_line', minimum=1) or 1
        end_line = self._optional_int(arguments, 'end_line', minimum=start_line) or len(lines) or 1
        selected = lines[start_line - 1 : end_line]
        numbered = '\n'.join(f'{index}\t{line}' for index, line in enumerate(selected, start=start_line))

        return RuntimeToolResult(
            name='read_file',
            success=True,
            output=self._truncate(numbered if numbered else ''),
            metadata={
                'path': str(path.relative_to(self.root)),
                'start_line': start_line,
                'end_line': min(end_line, len(lines)) if lines else 0,
                'line_count': len(lines),
            },
        )

    def _list_dir(self, arguments: dict[str, Any]) -> RuntimeToolResult:
        path = self._resolve_path(arguments.get('path', '.'))
        if not path.exists():
            raise RuntimeToolError(f'Path not found: {path.relative_to(self.root)}')
        if not path.is_dir():
            raise RuntimeToolError(f'Path is not a directory: {path.relative_to(self.root)}')
        max_entries = self._optional_int(arguments, 'max_entries', minimum=1) or 200
        entries = []
        for child in sorted(path.iterdir())[:max_entries]:
            marker = '[D]' if child.is_dir() else '[F]'
            entries.append(f'{marker} {child.relative_to(self.root)}')
        return RuntimeToolResult(
            name='list_dir',
            success=True,
            output='\n'.join(entries),
            metadata={'path': str(path.relative_to(self.root)), 'entry_count': len(entries)},
        )

    def _search_files(self, arguments: dict[str, Any]) -> RuntimeToolResult:
        pattern = self._require_string(arguments, 'pattern')
        search_root = self._resolve_path(arguments.get('path', '.'))
        if not search_root.exists():
            raise RuntimeToolError(f'Search path not found: {search_root.relative_to(self.root)}')
        max_results = self._optional_int(arguments, 'max_results', minimum=1) or 50

        if which('rg'):
            command = [
                'rg',
                '-n',
                '--no-heading',
                '--color',
                'never',
                '--max-count',
                str(max_results),
                pattern,
                str(search_root),
            ]
            result = subprocess.run(command, capture_output=True, text=True, cwd=self.root, check=False)
            if result.returncode not in (0, 1):
                raise RuntimeToolError(result.stderr.strip() or 'ripgrep search failed')
            output = result.stdout.strip()
        else:
            matches: list[str] = []
            for path in sorted(search_root.rglob('*')):
                if not path.is_file():
                    continue
                try:
                    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
                        if pattern in line:
                            matches.append(f'{path.relative_to(self.root)}:{line_number}:{line}')
                            if len(matches) >= max_results:
                                break
                    if len(matches) >= max_results:
                        break
                except UnicodeDecodeError:
                    continue
            output = '\n'.join(matches)

        return RuntimeToolResult(
            name='search_files',
            success=True,
            output=self._truncate(output),
            metadata={
                'path': str(search_root.relative_to(self.root)),
                'pattern': pattern,
                'max_results': max_results,
            },
        )

    def _edit_file(self, arguments: dict[str, Any]) -> RuntimeToolResult:
        path = self._resolve_path(self._require_string(arguments, 'path'))
        new_text = self._require_string(arguments, 'new_text')
        old_text = arguments.get('old_text')
        if old_text is not None and not isinstance(old_text, str):
            raise RuntimeToolError('old_text must be a string when provided.')

        if not path.exists():
            if old_text:
                raise RuntimeToolError(f'Cannot replace text in missing file: {path.relative_to(self.root)}')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_text, encoding='utf-8')
            return RuntimeToolResult(
                name='edit_file',
                success=True,
                output=f'Created {path.relative_to(self.root)}',
                metadata={'path': str(path.relative_to(self.root)), 'created': True},
            )

        if not path.is_file():
            raise RuntimeToolError(f'Path is not a file: {path.relative_to(self.root)}')
        if not old_text:
            raise RuntimeToolError('old_text is required when editing an existing file.')

        current = path.read_text(encoding='utf-8')
        match_count = current.count(old_text)
        if match_count == 0:
            raise RuntimeToolError('old_text was not found in the target file.')
        if match_count > 1:
            raise RuntimeToolError('old_text matched multiple locations; provide a more specific snippet.')

        path.write_text(current.replace(old_text, new_text, 1), encoding='utf-8')
        return RuntimeToolResult(
            name='edit_file',
            success=True,
            output=f'Updated {path.relative_to(self.root)}',
            metadata={'path': str(path.relative_to(self.root)), 'created': False},
        )

    def _run_shell_command(self, arguments: dict[str, Any]) -> RuntimeToolResult:
        command = self._require_string(arguments, 'command')
        timeout = self._optional_int(arguments, 'timeout_seconds', minimum=1) or 20
        return self._run_command(
            name='run_shell_command',
            command=command,
            timeout=timeout,
        )

    def _git_status(self, arguments: dict[str, Any]) -> RuntimeToolResult:
        del arguments
        self._ensure_git_repo()
        return self._run_command(
            name='git_status',
            command='git status --short --branch',
            timeout=20,
        )

    def _git_diff(self, arguments: dict[str, Any]) -> RuntimeToolResult:
        self._ensure_git_repo()
        staged = self._optional_bool(arguments, 'staged') or False
        path = arguments.get('path')
        if path is not None and not isinstance(path, str):
            raise RuntimeToolError('path must be a string when provided.')
        command_parts = ['git', 'diff']
        if staged:
            command_parts.append('--cached')
        if isinstance(path, str) and path:
            resolved = self._resolve_path(path)
            command_parts.extend(['--', str(resolved.relative_to(self.root))])
        return self._run_command(
            name='git_diff',
            command=' '.join(command_parts),
            timeout=20,
        )

    def _run_tests(self, arguments: dict[str, Any]) -> RuntimeToolResult:
        command = arguments.get('command')
        if command is not None and not isinstance(command, str):
            raise RuntimeToolError('command must be a string when provided.')
        timeout = self._optional_int(arguments, 'timeout_seconds', minimum=1) or 120
        return self._run_command(
            name='run_tests',
            command=command or 'python3 -m unittest discover -s tests -v',
            timeout=timeout,
        )

    def _run_build(self, arguments: dict[str, Any]) -> RuntimeToolResult:
        command = arguments.get('command')
        if command is not None and not isinstance(command, str):
            raise RuntimeToolError('command must be a string when provided.')
        timeout = self._optional_int(arguments, 'timeout_seconds', minimum=1) or 120
        return self._run_command(
            name='run_build',
            command=command or 'python3 -m compileall src',
            timeout=timeout,
        )

    def _run_command(self, name: str, command: str, timeout: int) -> RuntimeToolResult:
        try:
            result = subprocess.run(
                [self.shell, '-lc', command],
                capture_output=True,
                text=True,
                cwd=self.root,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeToolError(f'Shell command timed out after {timeout} seconds.') from error

        combined = '\n'.join(
            part for part in (result.stdout.strip(), result.stderr.strip()) if part
        )
        return RuntimeToolResult(
            name=name,
            success=result.returncode == 0,
            output=self._truncate(combined),
            metadata={'exit_code': result.returncode, 'command': command, 'timeout_seconds': timeout},
            error=None if result.returncode == 0 else f'Command exited with status {result.returncode}.',
        )

    def _ensure_git_repo(self) -> None:
        result = subprocess.run(
            ['git', 'rev-parse', '--is-inside-work-tree'],
            capture_output=True,
            text=True,
            cwd=self.root,
            check=False,
        )
        if result.returncode != 0 or result.stdout.strip() != 'true':
            raise RuntimeToolError('Workspace root is not inside a git work tree.')

    def _resolve_path(self, raw_path: Any) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise RuntimeToolError('path must be a non-empty string.')
        candidate = Path(raw_path)
        resolved = candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
        if not resolved.is_relative_to(self.root):
            raise RuntimeToolError('path must stay inside the workspace root.')
        return resolved

    @staticmethod
    def _require_string(arguments: dict[str, Any], key: str) -> str:
        value = arguments.get(key)
        if not isinstance(value, str) or not value:
            raise RuntimeToolError(f'{key} must be a non-empty string.')
        return value

    @staticmethod
    def _optional_int(arguments: dict[str, Any], key: str, minimum: int = 0) -> int | None:
        if key not in arguments or arguments[key] is None:
            return None
        value = arguments[key]
        if not isinstance(value, int) or value < minimum:
            raise RuntimeToolError(f'{key} must be an integer >= {minimum}.')
        return value

    @staticmethod
    def _optional_bool(arguments: dict[str, Any], key: str) -> bool | None:
        if key not in arguments or arguments[key] is None:
            return None
        value = arguments[key]
        if not isinstance(value, bool):
            raise RuntimeToolError(f'{key} must be a boolean.')
        return value

    @staticmethod
    def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
        if len(text) <= limit:
            return text
        return f'{text[:limit]}\n...[truncated]'
