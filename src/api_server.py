"""Minimal stdlib HTTP REST API for the autoresearch control plane.

A local language model (or any HTTP client) can drive the full research loop
by calling these JSON endpoints instead of shelling out to the CLI.

Usage::

    python3 -m src.main api-server --root /path/to/autoresearch-macos --port 7331

Then from any LLM tool-use handler::

    GET  http://localhost:7331/health         → {"status": "ok", "root": "..."}
    GET  http://localhost:7331/status         → autoresearch_status()
    POST http://localhost:7331/setup          body: {"root": "..."}  (optional override)
    POST http://localhost:7331/isolate        body: {"branch": "autoresearch/apr08", "create": true}
    POST http://localhost:7331/baseline       body: {}
    POST http://localhost:7331/run            body: {"packet": {...} or "packet_path": "..."}
    POST http://localhost:7331/keep           body: {}
    POST http://localhost:7331/discard        body: {}
    POST http://localhost:7331/retry          body: {}
    POST http://localhost:7331/loop           body: {"packet": {...}, "iterations": 3}
    GET  http://localhost:7331/log            → parse_run_log metrics
    POST http://localhost:7331/parse-log      body: {"log_path": "run.log"}

All responses are JSON.  Errors return {"error": "...message..."} with a 4xx/5xx status.

The server is intentionally single-threaded and stateless between requests.
All persistent state lives in the autoresearch repo files.

Connection to the Rust harness
-------------------------------
The Rust ``claw`` binary (``rust/crates/rusty-claude-cli``) implements its own
agent loop backed by the Anthropic/OpenAI API.  This Python API server exposes
the *control plane* (state transitions, keep/discard, TSV logging) so the
``claw`` CLI—or any LLM with HTTP tool-use—can manage research sessions without
needing direct Python imports.

Typical split:
- ``claw`` (Rust + Claude API) → acts as the *manager*, calls this API to record
  decisions and advance the research state.
- Python worker (Ollama or OpenAI-compat) → acts as the *code editor*, called
  internally by ``POST /run``.
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .autoresearch_runner import parse_run_log, setup_autoresearch
from .autoresearch_worker import (
    AutoresearchExperimentPacket,
    autoresearch_status,
    discard_autoresearch_candidate,
    ensure_autoresearch_baseline,
    ensure_autoresearch_branch,
    keep_autoresearch_candidate,
    load_autoresearch_state,
    load_memory_events,
    summarize_memory_events,
    load_autoresearch_packet,
    loop_autoresearch,
    retry_autoresearch_candidate,
    run_autoresearch_packet,
)
from .llm_backend import LLMBackend, OllamaBackend, OpenAICompatBackend, create_backend

DEFAULT_API_PORT = 7331
DEFAULT_API_HOST = '127.0.0.1'


class AutoresearchAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler that routes to autoresearch control-plane functions."""

    # server.root and server.backend are injected by AutoresearchAPIServer
    server: 'AutoresearchAPIServer'

    # ------------------------------------------------------------------ GET --

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip('/')
        query = parse_qs(parsed.query)
        if path == '/health':
            self._ok({'status': 'ok', 'root': str(self.server.root)})
        elif path == '/status':
            self._run(lambda _: autoresearch_status(self.server.root))
        elif path == '/memory':
            self._run(lambda _: {'memory': load_memory_events(self.server.root, limit=self._query_limit(query))})
        elif path == '/memory-summary':
            self._run(lambda _: summarize_memory_events(self.server.root, limit=self._query_limit(query)))
        elif path == '/log':
            log_file = self.server.root / 'run.log'
            self._run(lambda _: parse_run_log(log_file).to_dict())
        else:
            self._error(404, f'Unknown endpoint: {path}')

    # ----------------------------------------------------------------- POST --

    def do_POST(self) -> None:
        path = self.path.split('?')[0].rstrip('/')
        body = self._read_body()

        if path == '/setup':
            root = body.get('root') or str(self.server.root)
            self._run(lambda _: setup_autoresearch(root).to_dict())

        elif path == '/isolate':
            branch = body.get('branch')
            create = bool(body.get('create', False))
            from_ref = body.get('from_ref', 'HEAD')
            self._run(lambda _: ensure_autoresearch_branch(
                root=self.server.root,
                branch=branch,
                create=create,
                from_ref=from_ref,
            ))

        elif path == '/baseline':
            command = str(body.get('command', 'uv run train.py > run.log 2>&1'))
            log_path = str(body.get('log_path', 'run.log'))
            timeout_seconds = int(body.get('timeout_seconds', 600))
            results_tsv = str(body.get('results_tsv', 'results.tsv'))
            self._run(
                lambda _: ensure_autoresearch_baseline(
                    root=self.server.root,
                    command=command,
                    log_path=log_path,
                    results_tsv=results_tsv,
                    timeout_seconds=timeout_seconds,
                )
            )

        elif path == '/run':
            state = load_autoresearch_state(self.server.root)
            if state.pending_experiment is not None:
                self._error(409, 'experiment already pending', extra={'state': 'pending'})
                return
            packet = self._resolve_packet(body)
            backend = self._resolve_backend(body)
            self._run(lambda _: run_autoresearch_packet(
                packet,
                root=self.server.root,
                trace=bool(body.get('trace', False)),
                backend=backend,
            ))

        elif path == '/keep':
            rationale = body.get('rationale', '')
            self._run(lambda _: keep_autoresearch_candidate(root=self.server.root, rationale=str(rationale)))

        elif path == '/discard':
            rationale = body.get('rationale', '')
            self._run(lambda _: discard_autoresearch_candidate(root=self.server.root, rationale=str(rationale)))

        elif path == '/retry':
            self._run(lambda _: retry_autoresearch_candidate(root=self.server.root))

        elif path == '/loop':
            packet = self._resolve_packet(body)
            iterations = int(body.get('iterations', 1))
            retry_limit = int(body.get('retry_limit', 1))
            allow_any_branch = bool(body.get('allow_any_branch', False))
            backend = self._resolve_backend(body)
            self._run(lambda _: loop_autoresearch(
                packet,
                root=self.server.root,
                iterations=iterations,
                retry_limit=retry_limit,
                require_isolated_branch=not allow_any_branch,
                trace=bool(body.get('trace', False)),
                backend=backend,
            ))

        elif path == '/parse-log':
            log_path = body.get('log_path', 'run.log')
            target = Path(log_path)
            if not target.is_absolute():
                target = self.server.root / target
            self._run(lambda _: parse_run_log(target).to_dict())

        else:
            self._error(404, f'Unknown endpoint: {path}')

    # ------------------------------------------------------------ helpers --

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get('Content-Length', 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode('utf-8', errors='replace')
        if not raw.strip():
            return {}
        try:
            payload = json.loads(raw)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _resolve_packet(self, body: dict[str, Any]) -> AutoresearchExperimentPacket:
        if 'packet_path' in body:
            path = Path(str(body['packet_path']))
            if not path.is_absolute():
                path = self.server.root / path
            return load_autoresearch_packet(path)
        if 'packet' in body and isinstance(body['packet'], dict):
            return AutoresearchExperimentPacket.from_dict(body['packet'])
        # Fall back to the standard smoke packet sitting next to the node
        default = self.server.root / 'experiment.demo.json'
        if default.exists():
            return load_autoresearch_packet(default)
        raise ValueError(
            'No packet provided. Supply "packet": {...} or "packet_path": "..." in the request body.'
        )

    def _resolve_backend(self, body: dict[str, Any]) -> LLMBackend:
        """Build a backend from request body overrides, falling back to the server default."""
        kind = body.get('backend_kind')
        if kind is not None:
            return create_backend(
                backend_kind=str(kind),
                model=body.get('model'),
                host=body.get('host'),
                api_key=body.get('api_key'),
            )
        # Apply per-request model/host overrides on top of the server default
        server_backend = self.server.backend
        model_override = body.get('model')
        host_override = body.get('host')
        if model_override or host_override:
            if isinstance(server_backend, OpenAICompatBackend):
                return OpenAICompatBackend(
                    model=model_override or server_backend.model,
                    host=host_override or server_backend.host,
                    api_key=server_backend.api_key,
                )
            return OllamaBackend(
                model=model_override or server_backend.model,
                host=host_override or server_backend.host,
            )
        return server_backend

    def _query_limit(self, query: dict[str, list[str]]) -> int | None:
        raw_values = query.get('limit')
        if not raw_values:
            return None
        raw = raw_values[0].strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError as error:
            raise ValueError('limit query parameter must be an integer') from error
        if value < 0:
            raise ValueError('limit query parameter must be >= 0')
        return value

    def _run(self, fn: Any) -> None:
        try:
            result = fn(None)
            self._ok(result)
        except Exception as exc:
            self._error(500, str(exc))

    def _ok(self, data: Any) -> None:
        body = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, message: str, extra: dict[str, Any] | None = None) -> None:
        payload = {'error': message}
        if extra:
            payload.update(extra)
        body = json.dumps(payload).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N802
        # Override to suppress the default noisy access log; keep it simple.
        print(f'[api] {self.address_string()} {fmt % args}')


class AutoresearchAPIServer(HTTPServer):
    """HTTPServer subclass that holds shared server-level state."""

    def __init__(
        self,
        root: Path,
        backend: LLMBackend,
        host: str = DEFAULT_API_HOST,
        port: int = DEFAULT_API_PORT,
    ) -> None:
        self.root = root.resolve()
        self.backend = backend
        super().__init__((host, port), AutoresearchAPIHandler)


def run_api_server(
    root: Path,
    backend: LLMBackend,
    host: str = DEFAULT_API_HOST,
    port: int = DEFAULT_API_PORT,
) -> None:
    """Start the autoresearch REST API and block until interrupted."""
    server = AutoresearchAPIServer(root=root, backend=backend, host=host, port=port)
    print(f'[api] autoresearch API listening on http://{host}:{port}  root={root}')
    print(f'[api] backend={backend}')
    print('[api] Press Ctrl-C to stop.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[api] shutting down.')
        server.server_close()
