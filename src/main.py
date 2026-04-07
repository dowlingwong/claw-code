from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .autoresearch_runner import parse_run_log, setup_autoresearch
from .autoresearch_worker import (
    autoresearch_status,
    discard_autoresearch_candidate,
    ensure_autoresearch_baseline,
    ensure_autoresearch_branch,
    keep_autoresearch_candidate,
    loop_autoresearch,
    run_autoresearch_packet,
)
from .bootstrap_graph import build_bootstrap_graph
from .command_graph import build_command_graph
from .commands import execute_command, get_command, get_commands, render_command_index
from .agent_loop import DEFAULT_AGENT_SYSTEM_PROMPT, run_agent_task
from .llm_backend import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL, LLMBackendError
from .direct_modes import run_deep_link, run_direct_connect
from .parity_audit import run_parity_audit
from .permissions import ToolPermissionContext
from .port_manifest import build_port_manifest
from .query_engine import QueryEnginePort
from .remote_runtime import run_remote_mode, run_ssh_mode, run_teleport_mode
from .runtime import PortRuntime
from .session_store import load_session
from .setup import run_setup
from .task import DEFAULT_SYSTEM_PROMPT, run_local_task
from .worker_api import close_worker, create_worker, list_workers, load_worker, resume_worker, run_worker
from .tool_pool import assemble_tool_pool
from .tools import execute_tool, get_tool, get_tools, render_tool_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Python porting workspace for the Claude Code rewrite effort')
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('summary', help='render a Markdown summary of the Python porting workspace')
    subparsers.add_parser('manifest', help='print the current Python workspace manifest')
    subparsers.add_parser('parity-audit', help='compare the Python workspace against the local ignored TypeScript archive when available')
    subparsers.add_parser('setup-report', help='render the startup/prefetch setup report')
    subparsers.add_parser('command-graph', help='show command graph segmentation')
    subparsers.add_parser('tool-pool', help='show assembled tool pool with default settings')
    subparsers.add_parser('bootstrap-graph', help='show the mirrored bootstrap/runtime graph stages')
    list_parser = subparsers.add_parser('subsystems', help='list the current Python modules in the workspace')
    list_parser.add_argument('--limit', type=int, default=32)

    commands_parser = subparsers.add_parser('commands', help='list mirrored command entries from the archived snapshot')
    commands_parser.add_argument('--limit', type=int, default=20)
    commands_parser.add_argument('--query')
    commands_parser.add_argument('--no-plugin-commands', action='store_true')
    commands_parser.add_argument('--no-skill-commands', action='store_true')

    tools_parser = subparsers.add_parser('tools', help='list mirrored tool entries from the archived snapshot')
    tools_parser.add_argument('--limit', type=int, default=20)
    tools_parser.add_argument('--query')
    tools_parser.add_argument('--simple-mode', action='store_true')
    tools_parser.add_argument('--no-mcp', action='store_true')
    tools_parser.add_argument('--deny-tool', action='append', default=[])
    tools_parser.add_argument('--deny-prefix', action='append', default=[])

    route_parser = subparsers.add_parser('route', help='route a prompt across mirrored command/tool inventories')
    route_parser.add_argument('prompt')
    route_parser.add_argument('--limit', type=int, default=5)

    bootstrap_parser = subparsers.add_parser('bootstrap', help='build a runtime-style session report from the mirrored inventories')
    bootstrap_parser.add_argument('prompt')
    bootstrap_parser.add_argument('--limit', type=int, default=5)

    loop_parser = subparsers.add_parser('turn-loop', help='run a small stateful turn loop for the mirrored runtime')
    loop_parser.add_argument('prompt')
    loop_parser.add_argument('--limit', type=int, default=5)
    loop_parser.add_argument('--max-turns', type=int, default=3)
    loop_parser.add_argument('--structured-output', action='store_true')

    flush_parser = subparsers.add_parser('flush-transcript', help='persist and flush a temporary session transcript')
    flush_parser.add_argument('prompt')

    load_session_parser = subparsers.add_parser('load-session', help='load a previously persisted session')
    load_session_parser.add_argument('session_id')

    remote_parser = subparsers.add_parser('remote-mode', help='simulate remote-control runtime branching')
    remote_parser.add_argument('target')
    ssh_parser = subparsers.add_parser('ssh-mode', help='simulate SSH runtime branching')
    ssh_parser.add_argument('target')
    teleport_parser = subparsers.add_parser('teleport-mode', help='simulate teleport runtime branching')
    teleport_parser.add_argument('target')
    direct_parser = subparsers.add_parser('direct-connect-mode', help='simulate direct-connect runtime branching')
    direct_parser.add_argument('target')
    deep_link_parser = subparsers.add_parser('deep-link-mode', help='simulate deep-link runtime branching')
    deep_link_parser.add_argument('target')

    show_command = subparsers.add_parser('show-command', help='show one mirrored command entry by exact name')
    show_command.add_argument('name')
    show_tool = subparsers.add_parser('show-tool', help='show one mirrored tool entry by exact name')
    show_tool.add_argument('name')

    exec_command_parser = subparsers.add_parser('exec-command', help='execute a mirrored command shim by exact name')
    exec_command_parser.add_argument('name')
    exec_command_parser.add_argument('prompt')

    exec_tool_parser = subparsers.add_parser('exec-tool', help='execute a mirrored tool shim by exact name')
    exec_tool_parser.add_argument('name')
    exec_tool_parser.add_argument('payload')

    chat_parser = subparsers.add_parser('chat', help='send a prompt to a local Ollama model')
    chat_parser.add_argument('prompt', nargs='?')
    chat_parser.add_argument('--model', default=DEFAULT_OLLAMA_MODEL)
    chat_parser.add_argument('--host', default=DEFAULT_OLLAMA_HOST)
    chat_parser.add_argument('--system')

    agent_parser = subparsers.add_parser('agent', help='run the local Ollama agent loop with executable workspace tools')
    agent_parser.add_argument('prompt', nargs='?')
    agent_parser.add_argument('--model', default=DEFAULT_OLLAMA_MODEL)
    agent_parser.add_argument('--host', default=DEFAULT_OLLAMA_HOST)
    agent_parser.add_argument('--system')
    agent_parser.add_argument('--max-turns', type=int, default=8)
    agent_parser.add_argument('--deny-tool', action='append', default=[])
    agent_parser.add_argument('--deny-prefix', action='append', default=[])
    agent_parser.add_argument('--trace', action='store_true')

    worker_parser = subparsers.add_parser('worker', help='manage structured local Ollama workers')
    worker_subparsers = worker_parser.add_subparsers(dest='worker_command', required=True)

    worker_create = worker_subparsers.add_parser('create', help='create a worker record')
    worker_create.add_argument('--root', default='.')
    worker_create.add_argument('--model', default=DEFAULT_OLLAMA_MODEL)
    worker_create.add_argument('--host', default=DEFAULT_OLLAMA_HOST)

    worker_subparsers.add_parser('list', help='list stored worker records')

    worker_run = worker_subparsers.add_parser('run', help='run a task packet on a worker')
    worker_run.add_argument('worker_id')
    worker_run.add_argument('--packet', required=True)
    worker_run.add_argument('--trace', action='store_true')

    worker_status = worker_subparsers.add_parser('status', help='show worker state')
    worker_status.add_argument('worker_id')

    worker_resume = worker_subparsers.add_parser('resume', help='rerun the last stored task packet')
    worker_resume.add_argument('worker_id')
    worker_resume.add_argument('--trace', action='store_true')

    worker_close = worker_subparsers.add_parser('close', help='close a worker record')
    worker_close.add_argument('worker_id')

    autoresearch_parser = subparsers.add_parser('autoresearch', help='project-specific control plane for autoresearch experiments')
    autoresearch_subparsers = autoresearch_parser.add_subparsers(dest='autoresearch_command', required=True)

    autoresearch_setup = autoresearch_subparsers.add_parser('setup', help='verify autoresearch repo state and initialize results.tsv')
    autoresearch_setup.add_argument('--root')
    autoresearch_setup.add_argument('--no-init-results', action='store_true')

    autoresearch_run = autoresearch_subparsers.add_parser('run', help='run one autoresearch experiment packet through the worker + experiment runner')
    autoresearch_run.add_argument('--root')
    autoresearch_run.add_argument('--packet', required=True)
    autoresearch_run.add_argument('--model', default=DEFAULT_OLLAMA_MODEL)
    autoresearch_run.add_argument('--host', default=DEFAULT_OLLAMA_HOST)
    autoresearch_run.add_argument('--trace', action='store_true')

    autoresearch_status_parser = autoresearch_subparsers.add_parser('status', help='show current autoresearch control-plane state')
    autoresearch_status_parser.add_argument('--root')

    autoresearch_baseline = autoresearch_subparsers.add_parser('baseline', help='record the baseline experiment if results.tsv is still empty')
    autoresearch_baseline.add_argument('--root')
    autoresearch_baseline.add_argument('--command', dest='train_command', default='uv run train.py > run.log 2>&1')
    autoresearch_baseline.add_argument('--log-path', default='run.log')
    autoresearch_baseline.add_argument('--results-tsv', default='results.tsv')
    autoresearch_baseline.add_argument('--timeout-seconds', type=int, default=600)

    autoresearch_keep = autoresearch_subparsers.add_parser('keep', help='finalize the pending autoresearch candidate as keep')
    autoresearch_keep.add_argument('--root')

    autoresearch_discard = autoresearch_subparsers.add_parser('discard', help='finalize the pending autoresearch candidate as discard or crash')
    autoresearch_discard.add_argument('--root')

    autoresearch_isolate = autoresearch_subparsers.add_parser('isolate', help='create or switch to an autoresearch/<tag> branch')
    autoresearch_isolate.add_argument('--root')
    autoresearch_isolate.add_argument('--branch')
    autoresearch_isolate.add_argument('--create', action='store_true')
    autoresearch_isolate.add_argument('--from-ref', default='HEAD')

    autoresearch_loop = autoresearch_subparsers.add_parser('loop', help='run repeated autoresearch iterations with automatic baseline and keep/discard decisions')
    autoresearch_loop.add_argument('--root')
    autoresearch_loop.add_argument('--packet', required=True)
    autoresearch_loop.add_argument('--model', default=DEFAULT_OLLAMA_MODEL)
    autoresearch_loop.add_argument('--host', default=DEFAULT_OLLAMA_HOST)
    autoresearch_loop.add_argument('--iterations', type=int, default=1)
    autoresearch_loop.add_argument('--retry-limit', type=int, default=1)
    autoresearch_loop.add_argument('--allow-any-branch', action='store_true')
    autoresearch_loop.add_argument('--trace', action='store_true')

    autoresearch_parse_log = autoresearch_subparsers.add_parser('parse-log', help='parse a train.py run log into machine-readable metrics')
    autoresearch_parse_log.add_argument('--root')
    autoresearch_parse_log.add_argument('log_path')
    return parser


def resolve_chat_prompt(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.prompt:
        return args.prompt
    prompt = sys.stdin.read().strip()
    if prompt:
        return prompt
    command_name = getattr(args, 'command', 'command')
    parser.error(f'{command_name} requires a prompt argument or stdin input')
    raise AssertionError('unreachable')


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = build_port_manifest()
    if args.command == 'summary':
        print(QueryEnginePort(manifest).render_summary())
        return 0
    if args.command == 'manifest':
        print(manifest.to_markdown())
        return 0
    if args.command == 'parity-audit':
        print(run_parity_audit().to_markdown())
        return 0
    if args.command == 'setup-report':
        print(run_setup().as_markdown())
        return 0
    if args.command == 'command-graph':
        print(build_command_graph().as_markdown())
        return 0
    if args.command == 'tool-pool':
        print(assemble_tool_pool().as_markdown())
        return 0
    if args.command == 'bootstrap-graph':
        print(build_bootstrap_graph().as_markdown())
        return 0
    if args.command == 'subsystems':
        for subsystem in manifest.top_level_modules[: args.limit]:
            print(f'{subsystem.name}\t{subsystem.file_count}\t{subsystem.notes}')
        return 0
    if args.command == 'commands':
        if args.query:
            print(render_command_index(limit=args.limit, query=args.query))
        else:
            commands = get_commands(include_plugin_commands=not args.no_plugin_commands, include_skill_commands=not args.no_skill_commands)
            output_lines = [f'Command entries: {len(commands)}', '']
            output_lines.extend(f'- {module.name} — {module.source_hint}' for module in commands[: args.limit])
            print('\n'.join(output_lines))
        return 0
    if args.command == 'tools':
        if args.query:
            print(render_tool_index(limit=args.limit, query=args.query))
        else:
            permission_context = ToolPermissionContext.from_iterables(args.deny_tool, args.deny_prefix)
            tools = get_tools(simple_mode=args.simple_mode, include_mcp=not args.no_mcp, permission_context=permission_context)
            output_lines = [f'Tool entries: {len(tools)}', '']
            output_lines.extend(f'- {module.name} — {module.source_hint}' for module in tools[: args.limit])
            print('\n'.join(output_lines))
        return 0
    if args.command == 'route':
        matches = PortRuntime().route_prompt(args.prompt, limit=args.limit)
        if not matches:
            print('No mirrored command/tool matches found.')
            return 0
        for match in matches:
            print(f'{match.kind}\t{match.name}\t{match.score}\t{match.source_hint}')
        return 0
    if args.command == 'bootstrap':
        print(PortRuntime().bootstrap_session(args.prompt, limit=args.limit).as_markdown())
        return 0
    if args.command == 'turn-loop':
        results = PortRuntime().run_turn_loop(args.prompt, limit=args.limit, max_turns=args.max_turns, structured_output=args.structured_output)
        for idx, result in enumerate(results, start=1):
            print(f'## Turn {idx}')
            print(result.output)
            print(f'stop_reason={result.stop_reason}')
        return 0
    if args.command == 'flush-transcript':
        engine = QueryEnginePort.from_workspace()
        engine.submit_message(args.prompt)
        path = engine.persist_session()
        print(path)
        print(f'flushed={engine.transcript_store.flushed}')
        return 0
    if args.command == 'load-session':
        session = load_session(args.session_id)
        print(f'{session.session_id}\n{len(session.messages)} messages\nin={session.input_tokens} out={session.output_tokens}')
        return 0
    if args.command == 'remote-mode':
        print(run_remote_mode(args.target).as_text())
        return 0
    if args.command == 'ssh-mode':
        print(run_ssh_mode(args.target).as_text())
        return 0
    if args.command == 'teleport-mode':
        print(run_teleport_mode(args.target).as_text())
        return 0
    if args.command == 'direct-connect-mode':
        print(run_direct_connect(args.target).as_text())
        return 0
    if args.command == 'deep-link-mode':
        print(run_deep_link(args.target).as_text())
        return 0
    if args.command == 'show-command':
        module = get_command(args.name)
        if module is None:
            print(f'Command not found: {args.name}')
            return 1
        print('\n'.join([module.name, module.source_hint, module.responsibility]))
        return 0
    if args.command == 'show-tool':
        module = get_tool(args.name)
        if module is None:
            print(f'Tool not found: {args.name}')
            return 1
        print('\n'.join([module.name, module.source_hint, module.responsibility]))
        return 0
    if args.command == 'exec-command':
        result = execute_command(args.name, args.prompt)
        print(result.message)
        return 0 if result.handled else 1
    if args.command == 'exec-tool':
        result = execute_tool(args.name, args.payload)
        print(result.message)
        return 0 if result.handled else 1
    if args.command == 'chat':
        prompt = resolve_chat_prompt(args, parser)
        try:
            print(
                run_local_task(
                    prompt,
                    model=args.model,
                    host=args.host,
                    system_prompt=args.system if args.system is not None else DEFAULT_SYSTEM_PROMPT,
                )
            )
        except LLMBackendError as error:
            print(error, file=sys.stderr)
            return 1
        return 0
    if args.command == 'agent':
        prompt = resolve_chat_prompt(args, parser)
        permission_context = ToolPermissionContext.from_iterables(args.deny_tool, args.deny_prefix)
        try:
            result = run_agent_task(
                prompt,
                model=args.model,
                host=args.host,
                system_prompt=args.system if args.system is not None else DEFAULT_AGENT_SYSTEM_PROMPT,
                max_turns=args.max_turns,
                permission_context=permission_context,
                trace=args.trace,
            )
            if args.trace:
                for event in result.trace_events:
                    print(f'[trace] {event}', file=sys.stderr)
            print(result.content)
        except LLMBackendError as error:
            print(error, file=sys.stderr)
            return 1
        return 0
    if args.command == 'worker':
        try:
            if args.worker_command == 'list':
                workers = list_workers()
                print(json.dumps([worker.to_dict() for worker in workers], indent=2))
                return 0
            if args.worker_command == 'create':
                worker = create_worker(root=args.root, model=args.model, host=args.host)
                print(json.dumps(worker.to_dict(), indent=2))
                return 0
            if args.worker_command == 'run':
                worker = run_worker(args.worker_id, args.packet, trace=args.trace)
                print(json.dumps(worker.to_dict(), indent=2))
                return 0
            if args.worker_command == 'status':
                worker = load_worker(args.worker_id)
                print(json.dumps(worker.to_dict(), indent=2))
                return 0
            if args.worker_command == 'resume':
                worker = resume_worker(args.worker_id, trace=args.trace)
                print(json.dumps(worker.to_dict(), indent=2))
                return 0
            if args.worker_command == 'close':
                worker = close_worker(args.worker_id)
                print(json.dumps(worker.to_dict(), indent=2))
                return 0
        except Exception as error:
            print(error, file=sys.stderr)
            return 1
    if args.command == 'autoresearch':
        try:
            if args.autoresearch_command == 'setup':
                report = setup_autoresearch(
                    root=args.root,
                    initialize_results=not args.no_init_results,
                )
                print(json.dumps(report.to_dict(), indent=2))
                return 0
            if args.autoresearch_command == 'run':
                result = run_autoresearch_packet(
                    args.packet,
                    root=args.root,
                    model=args.model,
                    host=args.host,
                    trace=args.trace,
                )
                print(json.dumps(result, indent=2))
                return 0
            if args.autoresearch_command == 'status':
                print(json.dumps(autoresearch_status(root=args.root), indent=2))
                return 0
            if args.autoresearch_command == 'baseline':
                result = ensure_autoresearch_baseline(
                    root=args.root,
                    command=args.train_command,
                    log_path=args.log_path,
                    results_tsv=args.results_tsv,
                    timeout_seconds=args.timeout_seconds,
                )
                print(json.dumps(result, indent=2))
                return 0
            if args.autoresearch_command == 'keep':
                print(json.dumps(keep_autoresearch_candidate(root=args.root), indent=2))
                return 0
            if args.autoresearch_command == 'discard':
                print(json.dumps(discard_autoresearch_candidate(root=args.root), indent=2))
                return 0
            if args.autoresearch_command == 'isolate':
                result = ensure_autoresearch_branch(
                    root=args.root,
                    branch=args.branch,
                    create=args.create,
                    from_ref=args.from_ref,
                )
                print(json.dumps(result, indent=2))
                return 0
            if args.autoresearch_command == 'loop':
                result = loop_autoresearch(
                    args.packet,
                    root=args.root,
                    model=args.model,
                    host=args.host,
                    iterations=args.iterations,
                    retry_limit=args.retry_limit,
                    require_isolated_branch=not args.allow_any_branch,
                    trace=args.trace,
                )
                print(json.dumps(result, indent=2))
                return 0
            if args.autoresearch_command == 'parse-log':
                log_path = args.log_path
                if not Path(log_path).is_absolute() and args.root:
                    log_path = str(Path(args.root) / log_path)
                metrics = parse_run_log(log_path)
                print(json.dumps(metrics.to_dict(), indent=2))
                return 0
        except Exception as error:
            print(error, file=sys.stderr)
            return 1
    parser.error(f'unknown command: {args.command}')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
