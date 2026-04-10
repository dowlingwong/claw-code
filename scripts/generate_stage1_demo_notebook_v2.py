from __future__ import annotations

import json
import textwrap
from pathlib import Path


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(text).lstrip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": textwrap.dedent(text).lstrip("\n").splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells: list[dict] = []

    cells.append(
        markdown(
            """
            # Stage 1 Sudhir Demo Verification v2

            This notebook verifies a **governed experimentation harness** with four concrete claims:

            - a manager can supervise the full experiment lifecycle through the control plane
            - the worker is bounded to `train.py`
            - memory persists across runs and influences later proposals
            - decisions are made from returned metrics, including a discard path

            What is synthetic vs real:

            - **Synthetic stability demo**: the worker makes a real `train.py` edit, but the experiment command writes a fixed metric to keep the first proof stable.
            - **Realistic execution demo**: the worker edits `train.py`, then `python3 train.py` computes a deterministic validation metric from the edited code.

            By the end, the notebook shows: baseline creation, manager-guided keep/discard decisions, bounded worker behavior, structured memory reuse, and final persisted artifacts.
            """
        )
    )

    cells.append(
        markdown(
            """
            ## What I Need From You

            Before running the code cells, set the configuration values in the next cell:

            - `BACKEND_KIND`: usually `"ollama"`
            - `BACKEND_HOST`: usually `"http://localhost:11434"`
            - `MODEL_NAME`: the local worker model you want to use, for example `"qwen2.5-coder:7b"`

            The notebook expects the local worker backend to already be running.

            - For Ollama: make sure the model is pulled and `ollama serve` is available.
            - For an OpenAI-compatible local server: make sure the `/v1` endpoint is live.

            The notebook creates a temporary demo repo, starts the real control-plane API server against it, and stops the server at the end. The temporary workspace remains on disk until the kernel exits so you can inspect the artifacts.
            """
        )
    )

    cells.append(
        markdown(
            """
            ## 1. System Initialization

            This section loads helper functions, prepares a temporary demo node, starts the control plane, and records a **real deterministic baseline** through the API. No user input is needed here beyond the backend configuration from the next cell.
            """
        )
    )

    cells.append(
        code(
            r'''
            from __future__ import annotations

            import atexit
            import json
            import os
            import re
            import socket
            import subprocess
            import sys
            import textwrap
            import time
            from pathlib import Path
            from tempfile import TemporaryDirectory
            from urllib.error import HTTPError
            from urllib.request import Request, urlopen

            NOTEBOOK_DIR = Path.cwd().resolve()
            CLAW_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'notebooks' else NOTEBOOK_DIR
            if str(CLAW_ROOT) not in sys.path:
                sys.path.insert(0, str(CLAW_ROOT))

            from src.demo_verification import (
                build_diff_summary,
                extract_metric_from_text,
                get_changed_files,
                snapshot_workspace_sources,
                tail_lines,
            )

            TEMP_DIR: TemporaryDirectory[str] | None = None
            API_PROC: subprocess.Popen[str] | None = None
            WORKSPACE_ROOT: Path | None = None
            NODE_ROOT: Path | None = None
            API_BASE: str | None = None


            def print_json(title: str, payload: object) -> None:
                print(f"\n=== {title} ===")
                print(json.dumps(payload, indent=2, sort_keys=True))


            def pick_free_port() -> int:
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    return int(sock.getsockname()[1])


            def http_get_json(url: str, timeout: int = 30) -> dict:
                with urlopen(url, timeout=timeout) as response:
                    return json.loads(response.read().decode('utf-8'))


            def http_post_json(url: str, payload: dict, timeout: int = 300) -> tuple[int, dict]:
                request = Request(
                    url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method='POST',
                )
                try:
                    with urlopen(request, timeout=timeout) as response:
                        return response.status, json.loads(response.read().decode('utf-8'))
                except HTTPError as error:
                    body = error.read().decode('utf-8', errors='replace').strip()
                    parsed = json.loads(body) if body else {'error': str(error)}
                    return error.code, parsed


            def run_cmd(
                args: list[str],
                cwd: Path,
                env: dict[str, str] | None = None,
                timeout: int = 30,
            ) -> subprocess.CompletedProcess[str]:
                result = subprocess.run(
                    args,
                    cwd=cwd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"command failed ({result.returncode}): {' '.join(args)}\n"
                        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                    )
                return result


            def demo_train_source() -> str:
                return textwrap.dedent(
                    """\
                    from __future__ import annotations

                    CALIBRATION_POINTS = [0.18, 0.12, 0.05, -0.02]
                    TARGET_BIAS = 0.140
                    MODEL_BIAS = 0.180


                    def compute_val_bpb() -> float:
                        squared_error = sum(
                            ((point + MODEL_BIAS) - (point + TARGET_BIAS)) ** 2 for point in CALIBRATION_POINTS
                        ) / len(CALIBRATION_POINTS)
                        return 0.950000 + squared_error


                    def main() -> None:
                        val_bpb = compute_val_bpb()
                        print(f"val_bpb:          {val_bpb:.6f}")
                        print("training_seconds: 0.2")
                        print("total_seconds:    0.3")
                        print("peak_vram_mb:     128.0")
                        print("mfu_percent:      1.0")
                        print("total_tokens_M:   0.001")
                        print("num_steps:        4")
                        print("num_params_M:     0.001")
                        print("depth:            1")


                    if __name__ == "__main__":
                        main()
                    """
                )


            def create_demo_repo(node_root: Path) -> None:
                node_root.mkdir(parents=True, exist_ok=True)
                (node_root / 'train.py').write_text(demo_train_source(), encoding='utf-8')
                (node_root / 'prepare.py').write_text(
                    'from pathlib import Path\n\nprint("prepare stub", Path.cwd())\n',
                    encoding='utf-8',
                )
                (node_root / 'program.md').write_text(
                    textwrap.dedent(
                        """\
                        # Demo Program

                        This node is a deterministic stage-one demo target.
                        The worker may only edit `train.py`.
                        The control plane records lifecycle, memory, and results.
                        """
                    ),
                    encoding='utf-8',
                )
                (node_root / 'README.md').write_text(
                    textwrap.dedent(
                        """\
                        # Stage 1 Demo Node

                        `train.py` computes a deterministic validation metric from `MODEL_BIAS`.
                        Lower `val_bpb` is better.
                        """
                    ),
                    encoding='utf-8',
                )
                (node_root / '.gitignore').write_text(
                    '\n'.join(
                        [
                            '.autoresearch_state.json',
                            'experiment_memory.jsonl',
                            'results.tsv',
                            'run.log',
                            '.port_workers/',
                            '__pycache__/',
                            '*.pyc',
                            '',
                        ]
                    ),
                    encoding='utf-8',
                )
                run_cmd(['git', 'init'], cwd=node_root)
                run_cmd(['git', 'config', 'user.email', 'demo@example.com'], cwd=node_root)
                run_cmd(['git', 'config', 'user.name', 'Stage 1 Demo'], cwd=node_root)
                run_cmd(['git', 'add', '.'], cwd=node_root)
                run_cmd(['git', 'commit', '-m', 'init demo node'], cwd=node_root)


            def check_backend_ready(kind: str, host: str) -> dict:
                base = host.rstrip('/')
                url = (
                    f"{base}/api/tags"
                    if kind == 'ollama'
                    else (f"{base}/v1/models" if not base.endswith('/v1') else f"{base}/models")
                )
                with urlopen(url, timeout=10) as response:
                    payload = json.loads(response.read().decode('utf-8'))
                return {
                    'backend_kind': kind,
                    'backend_host': host,
                    'probe_url': url,
                    'reachable': True,
                    'payload_preview_keys': sorted(payload.keys())[:8],
                }


            def start_api_server(
                node_root: Path,
                port: int,
                backend_kind: str,
                model_name: str,
                backend_host: str,
            ) -> tuple[subprocess.Popen[str], str]:
                env = os.environ.copy()
                env['PYTHONPATH'] = str(CLAW_ROOT)
                command = [
                    sys.executable,
                    '-m',
                    'src.main',
                    'api-server',
                    '--root',
                    str(node_root),
                    '--port',
                    str(port),
                    '--listen',
                    '127.0.0.1',
                    '--backend',
                    backend_kind,
                    '--model',
                    model_name,
                    '--host',
                    backend_host,
                ]
                proc = subprocess.Popen(
                    command,
                    cwd=CLAW_ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                api_base = f'http://127.0.0.1:{port}'
                deadline = time.time() + 10
                last_error = ''
                while time.time() < deadline:
                    if proc.poll() is not None:
                        stderr = proc.stderr.read() if proc.stderr else ''
                        stdout = proc.stdout.read() if proc.stdout else ''
                        raise RuntimeError(
                            f'API server exited early.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}'
                        )
                    try:
                        payload = http_get_json(f'{api_base}/health', timeout=2)
                        if payload.get('status') == 'ok':
                            return proc, api_base
                    except Exception as error:  # noqa: BLE001
                        last_error = str(error)
                        time.sleep(0.2)
                raise RuntimeError(f'API server did not become healthy: {last_error}')


            def stop_api_server() -> None:
                global API_PROC
                if API_PROC is None:
                    return
                if API_PROC.poll() is None:
                    API_PROC.terminate()
                    try:
                        API_PROC.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        API_PROC.kill()
                        API_PROC.wait(timeout=5)
                API_PROC = None


            def synthetic_train_command(val_bpb: float) -> str:
                return textwrap.dedent(
                    f"""\
                    python3 - <<'PY' > run.log 2>&1
                    print("val_bpb:          {val_bpb:.6f}")
                    print("training_seconds: 0.1")
                    print("total_seconds:    0.1")
                    print("peak_vram_mb:     64.0")
                    print("mfu_percent:      0.5")
                    print("total_tokens_M:   0.0001")
                    print("num_steps:        1")
                    print("num_params_M:     0.0001")
                    print("depth:            1")
                    PY
                    """
                )


            def real_train_command() -> str:
                return 'python3 train.py > run.log 2>&1'


            def build_synthetic_packet(val_bpb: float) -> dict:
                return {
                    'objective': 'Modify train.py by adding a smoke-test comment so the manager can prove the bounded worker path.',
                    'description': 'Add smoke-test comment for synthetic stability proof',
                    'train_command': synthetic_train_command(val_bpb),
                    'timeout_seconds': 20,
                    'syntax_check_command': 'python3 -m py_compile train.py',
                    'results_tsv': 'results.tsv',
                    'log_path': 'run.log',
                }


            def build_real_packet(model_bias: float) -> dict:
                return {
                    'objective': (
                        f'Modify train.py by setting MODEL_BIAS = {model_bias:.3f} '
                        'so the deterministic validation metric moves closer to TARGET_BIAS.'
                    ),
                    'description': f'Set MODEL_BIAS = {model_bias:.3f} for deterministic evaluation demo',
                    'train_command': real_train_command(),
                    'timeout_seconds': 20,
                    'syntax_check_command': 'python3 -m py_compile train.py',
                    'results_tsv': 'results.tsv',
                    'log_path': 'run.log',
                }


            def current_best_metric(api_base: str) -> float:
                status = http_get_json(f'{api_base}/status')
                return float(status['state']['best_bpb'])


            def manager_cycle(api_base: str, packet: dict, reference_metric: float, memory_note: str = '') -> dict:
                run_status, run_result = http_post_json(f'{api_base}/run', {'packet': packet, 'trace': True})
                assert run_status == 200, run_result

                experiment = run_result.get('experiment') or {}
                candidate_metric = experiment.get('val_bpb')
                run_success = bool(experiment.get('success')) and candidate_metric is not None

                if not run_success:
                    recommendation = 'reject'
                    decision_endpoint = 'discard'
                    rationale = (
                        f"{memory_note}rejecting experiment because execution failed: "
                        f"{experiment.get('error') or run_result.get('error') or 'unknown error'}"
                    ).strip()
                else:
                    candidate_metric = float(candidate_metric)
                    delta = candidate_metric - float(reference_metric)
                    if delta < 0:
                        recommendation = 'keep'
                        decision_endpoint = 'keep'
                        rationale = (
                            f"{memory_note}candidate improved by {abs(delta):.6f} "
                            f"vs reference {reference_metric:.6f} "
                            f"(candidate {candidate_metric:.6f})."
                        ).strip()
                    else:
                        recommendation = 'discard'
                        decision_endpoint = 'discard'
                        rationale = (
                            f"{memory_note}candidate regressed by {delta:.6f} "
                            f"vs reference {reference_metric:.6f} "
                            f"(candidate {candidate_metric:.6f})."
                        ).strip()

                decision_status, decision_result = http_post_json(
                    f'{api_base}/{decision_endpoint}',
                    {'rationale': rationale},
                )
                assert decision_status == 200, decision_result

                status_after = http_get_json(f'{api_base}/status')
                memory_after = http_get_json(f'{api_base}/memory?limit=1')
                latest_memory_entry = (memory_after.get('memory') or [None])[-1]
                return {
                    'packet': packet,
                    'reference_metric': float(reference_metric),
                    'run_result': run_result,
                    'candidate_metric': float(candidate_metric) if candidate_metric is not None else None,
                    'recommendation': recommendation,
                    'decision_endpoint': decision_endpoint,
                    'rationale': rationale,
                    'decision': decision_result,
                    'status_after': status_after,
                    'memory_after_latest': latest_memory_entry,
                }


            def extract_model_biases(memory_entries: list[dict]) -> list[float]:
                pattern = re.compile(r'MODEL_BIAS\s*=\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+))')
                values: set[float] = set()
                for entry in memory_entries:
                    payload = entry.get('payload', {})
                    if not isinstance(payload, dict):
                        continue
                    texts = [payload.get('description', ''), payload.get('objective', '')]
                    packet = payload.get('packet')
                    if isinstance(packet, dict):
                        texts.extend([packet.get('description', ''), packet.get('objective', '')])
                    for text in texts:
                        if not isinstance(text, str):
                            continue
                        for match in pattern.finditer(text):
                            values.add(round(float(match.group(1)), 3))
                return sorted(values)


            def choose_next_model_bias(
                memory_entries: list[dict],
                candidates: tuple[float, ...] = (0.140, 0.120, 0.160),
            ) -> float:
                tried = {round(value, 3) for value in extract_model_biases(memory_entries)}
                for candidate in candidates:
                    if round(candidate, 3) not in tried:
                        return candidate
                raise RuntimeError(f'All candidate MODEL_BIAS values were already tried: {sorted(tried)}')


            atexit.register(stop_api_server)
            '''
        )
    )

    cells.append(
        markdown(
            """
            ## Configure the local worker backend

            This cell is the only one that needs your machine-specific information. Update the values if your worker model or host differs from the defaults. The cell also probes the backend so the notebook fails early with a clear error if the local model is not reachable.
            """
        )
    )

    cells.append(
        code(
            """
            BACKEND_KIND = 'ollama'
            BACKEND_HOST = 'http://localhost:11434'
            MODEL_NAME = 'qwen2.5-coder:7b'
            API_PORT = pick_free_port()

            backend_probe = check_backend_ready(BACKEND_KIND, BACKEND_HOST)
            print_json('backend_probe', backend_probe)
            print(f'Configured model: {MODEL_NAME}')
            print(f'Planned API port: {API_PORT}')
            """
        )
    )

    cells.append(
        markdown(
            """
            ## Create the temporary node, start the control plane, and record a real baseline

            This cell builds a fresh git repo for the demo, starts the actual API server against it, isolates the repo on an autoresearch branch, and records a **real deterministic baseline** through `POST /baseline`. No further input is needed from you.
            """
        )
    )

    cells.append(
        code(
            """
            TEMP_DIR = TemporaryDirectory(prefix='stage1-demo-')
            WORKSPACE_ROOT = Path(TEMP_DIR.name)
            NODE_ROOT = WORKSPACE_ROOT / 'demo_node'
            create_demo_repo(NODE_ROOT)

            API_PROC, API_BASE = start_api_server(
                node_root=NODE_ROOT,
                port=API_PORT,
                backend_kind=BACKEND_KIND,
                model_name=MODEL_NAME,
                backend_host=BACKEND_HOST,
            )

            status_clean = http_get_json(f'{API_BASE}/status')
            isolate_status, isolate_result = http_post_json(
                f'{API_BASE}/isolate',
                {'branch': 'autoresearch/stage1-demo', 'create': True},
            )
            assert isolate_status == 200, isolate_result
            baseline_status, baseline_result = http_post_json(
                f'{API_BASE}/baseline',
                {
                    'command': real_train_command(),
                    'log_path': 'run.log',
                    'timeout_seconds': 20,
                    'results_tsv': 'results.tsv',
                },
            )
            assert baseline_status == 200, baseline_result
            status_after_baseline = http_get_json(f'{API_BASE}/status')

            assert status_clean['state']['pending_experiment'] is None
            assert status_after_baseline['state']['pending_experiment'] is None

            print(f'workspace path: {WORKSPACE_ROOT}')
            print(f'node path: {NODE_ROOT}')
            print_json('initial_state_before_baseline', status_clean['state'])
            print_json('baseline_result', baseline_result)
            print_json('state_after_baseline', status_after_baseline['state'])
            print('confirmation no pending experiment:', status_after_baseline['state']['pending_experiment'] is None)
            """
        )
    )

    cells.append(
        markdown(
            """
            ## 2. API Contract Demonstration

            This cell calls the structured control-plane interface directly. It shows `GET /status`, `GET /memory`, and `GET /memory-summary`, then prints the key fields a manager would need: baseline metric, memory length, and last decision.
            """
        )
    )

    cells.append(
        code(
            """
            status_payload = http_get_json(f'{API_BASE}/status')
            memory_payload = http_get_json(f'{API_BASE}/memory?limit=20')
            memory_summary_payload = http_get_json(f'{API_BASE}/memory-summary?limit=20')

            print_json('GET /status', status_payload)
            print_json('GET /memory', memory_payload)
            print_json('GET /memory-summary', memory_summary_payload)
            print('baseline_bpb:', status_payload['state']['baseline_bpb'])
            print('memory_length:', len(memory_payload['memory']))
            print('last_decision:', memory_summary_payload.get('latest_decision'))
            """
        )
    )

    cells.append(
        markdown(
            """
            ## 3. Stable Demo Run (Baseline Proof)

            This cell executes one full manager cycle with a **synthetic** experiment command. The worker still performs a real edit to `train.py`, but the command writes a fixed metric so the lifecycle proof is stable and easy to verify. No extra input is needed.
            """
        )
    )

    cells.append(
        code(
            """
            stable_reference_metric = current_best_metric(API_BASE)
            STABLE_BEFORE_SNAPSHOT = snapshot_workspace_sources(NODE_ROOT)
            STABLE_PACKET = build_synthetic_packet(val_bpb=0.951000)
            STABLE_CYCLE = manager_cycle(API_BASE, STABLE_PACKET, reference_metric=stable_reference_metric)
            STABLE_AFTER_SNAPSHOT = snapshot_workspace_sources(NODE_ROOT)

            print_json('manager_proposal_stable', STABLE_PACKET)
            print('before metric:', f'{stable_reference_metric:.6f}')
            print('after metric:', f"{STABLE_CYCLE['candidate_metric']:.6f}")
            print('decision rationale:', STABLE_CYCLE['rationale'])
            print_json('stable_run_result', STABLE_CYCLE['run_result'])
            print_json('stable_memory_entry', STABLE_CYCLE['memory_after_latest'])
            assert STABLE_CYCLE['decision']['decision'] == 'keep'
            """
        )
    )

    cells.append(
        markdown(
            """
            ## 4. Bounded Worker Proof

            This cell proves worker scope with an explicit before/after source snapshot. The diff check focuses on the governed source files in the node and must assert that only `train.py` changed. No user input is needed.
            """
        )
    )

    cells.append(
        code(
            """
            stable_changed_files = get_changed_files(STABLE_BEFORE_SNAPSHOT, STABLE_AFTER_SNAPSHOT)
            stable_diff_summary = build_diff_summary(STABLE_BEFORE_SNAPSHOT, STABLE_AFTER_SNAPSHOT)

            assert stable_changed_files == ['train.py'], stable_changed_files
            print('changed_files:', stable_changed_files)
            print_json('git_style_diff_summary', stable_diff_summary)
            print('number_of_lines_changed:', stable_diff_summary[0]['changed_line_count'])
            print(stable_diff_summary[0]['diff'])
            """
        )
    )

    cells.append(
        markdown(
            """
            ## 5. Realistic Experiment Run

            This cell replaces the synthetic command with a **real deterministic evaluation script**: `python3 train.py > run.log 2>&1`. The manager proposes a concrete `MODEL_BIAS` change, the worker edits `train.py`, the command runs, and the notebook parses the returned metric from the real log output.
            """
        )
    )

    cells.append(
        code(
            """
            real_reference_metric = current_best_metric(API_BASE)
            REAL_PACKET = build_real_packet(model_bias=0.130)
            REAL_CYCLE = manager_cycle(API_BASE, REAL_PACKET, reference_metric=real_reference_metric)
            REAL_LOG_TEXT = (NODE_ROOT / 'run.log').read_text(encoding='utf-8', errors='replace')
            REAL_LOG_SNIPPET = tail_lines(REAL_LOG_TEXT, limit=12)
            REAL_PARSED_METRIC = extract_metric_from_text(REAL_LOG_TEXT)

            print_json('manager_proposal_realistic', REAL_PACKET)
            print('real command executed:', REAL_PACKET['train_command'])
            print('raw output snippet:')
            print(REAL_LOG_SNIPPET)
            print('parsed metric:', REAL_PARSED_METRIC)
            print(
                'comparison vs reference best:',
                {'before_best_metric': real_reference_metric, 'candidate_metric': REAL_CYCLE['candidate_metric']},
            )
            print_json('realistic_run_result', REAL_CYCLE['run_result'])
            assert REAL_CYCLE['decision']['decision'] == 'keep'
            assert REAL_PARSED_METRIC == REAL_CYCLE['candidate_metric']
            """
        )
    )

    cells.append(
        markdown(
            """
            ## 6. Decision Logic Transparency

            This cell computes the decision inputs explicitly: reference metric, candidate metric, delta, recommendation, and final manager decision. No extra information is needed from you.
            """
        )
    )

    cells.append(
        code(
            """
            decision_baseline_metric = REAL_CYCLE['reference_metric']
            decision_candidate_metric = REAL_CYCLE['candidate_metric']
            decision_delta = decision_candidate_metric - decision_baseline_metric
            expected_recommendation = 'keep' if decision_delta < 0 else 'discard'
            transparent_decision = {
                'baseline_metric': round(decision_baseline_metric, 6),
                'candidate_metric': round(decision_candidate_metric, 6),
                'delta': round(decision_delta, 6),
                'recommendation': expected_recommendation,
                'manager_decision': REAL_CYCLE['decision']['decision'],
            }
            print_json('decision_logic', transparent_decision)
            assert transparent_decision['recommendation'] == transparent_decision['manager_decision']
            """
        )
    )

    cells.append(
        markdown(
            """
            ## 7. Failure / Discard Path

            This cell runs a controlled regression. The manager proposes a worse `MODEL_BIAS`, the experiment returns a worse metric, and the manager calls `POST /discard`. The resulting memory entry should include a `failure_tag`. No additional input is required.
            """
        )
    )

    cells.append(
        code(
            """
            failure_reference_metric = current_best_metric(API_BASE)
            FAILURE_PACKET = build_real_packet(model_bias=0.250)
            FAILURE_CYCLE = manager_cycle(API_BASE, FAILURE_PACKET, reference_metric=failure_reference_metric)
            FAILURE_MEMORY_ENTRY = FAILURE_CYCLE['memory_after_latest']
            FAILURE_PAYLOAD = FAILURE_MEMORY_ENTRY['payload']

            print_json('failure_packet', FAILURE_PACKET)
            print('failure type:', FAILURE_PAYLOAD.get('failure_tag') or 'unknown')
            print('recommendation:', FAILURE_CYCLE['recommendation'])
            print('discard decision:', FAILURE_CYCLE['decision']['decision'])
            print_json('failure_memory_entry', FAILURE_MEMORY_ENTRY)
            assert FAILURE_CYCLE['decision']['decision'] == 'discard'
            assert FAILURE_PAYLOAD.get('failure_tag') == 'no_improvement'
            """
        )
    )

    cells.append(
        markdown(
            """
            ## 8. Memory Reuse

            This cell runs a second realistic iteration. The manager reads memory first, identifies previously tried `MODEL_BIAS` values, avoids repeating them, explains the choice, and proposes a different change. If you want to inspect the logic, the memory contents are printed before the new proposal is sent.
            """
        )
    )

    cells.append(
        code(
            """
            memory_before_second_run = http_get_json(f'{API_BASE}/memory?limit=30')['memory']
            tried_model_biases = extract_model_biases(memory_before_second_run)
            chosen_model_bias = choose_next_model_bias(memory_before_second_run, candidates=(0.140, 0.120, 0.160))
            memory_influence_explanation = (
                f'Memory showed prior MODEL_BIAS attempts {tried_model_biases}. '
                f'The manager avoided repeating those values and proposed MODEL_BIAS = {chosen_model_bias:.3f}.'
            )
            MEMORY_GUIDED_PACKET = build_real_packet(model_bias=chosen_model_bias)
            memory_guided_reference_metric = current_best_metric(API_BASE)
            MEMORY_GUIDED_CYCLE = manager_cycle(
                API_BASE,
                MEMORY_GUIDED_PACKET,
                reference_metric=memory_guided_reference_metric,
                memory_note=memory_influence_explanation + ' ',
            )

            print_json('memory_before_second_run', memory_before_second_run)
            print('memory influence explanation:', memory_influence_explanation)
            print_json('new_experiment_proposal', MEMORY_GUIDED_PACKET)
            print_json(
                'memory_guided_cycle',
                {
                    'reference_metric': MEMORY_GUIDED_CYCLE['reference_metric'],
                    'candidate_metric': MEMORY_GUIDED_CYCLE['candidate_metric'],
                    'recommendation': MEMORY_GUIDED_CYCLE['recommendation'],
                    'decision': MEMORY_GUIDED_CYCLE['decision'],
                },
            )
            assert round(chosen_model_bias, 3) not in {round(value, 3) for value in tried_model_biases}
            """
        )
    )

    cells.append(
        markdown(
            """
            ## 9. Memory Structure Inspection

            This cell prints one structured memory entry so you can inspect the manager-facing schema directly. It highlights the fields needed for auditability and reuse. No user input is needed.
            """
        )
    )

    cells.append(
        code(
            """
            all_memory_entries = http_get_json(f'{API_BASE}/memory?limit=50')['memory']
            inspected_memory_entry = next(entry for entry in reversed(all_memory_entries) if entry.get('event') == 'discard')
            inspected_payload = inspected_memory_entry['payload']
            structured_memory_view = {
                'experiment_id': inspected_payload.get('experiment_id'),
                'objective': (inspected_payload.get('packet') or {}).get('objective', inspected_payload.get('objective')),
                'code_change_summary': inspected_payload.get('code_change_summary'),
                'metrics': {
                    'val_bpb': inspected_payload.get('val_bpb'),
                    'success': (inspected_payload.get('experiment') or {}).get('success'),
                },
                'decision': inspected_payload.get('decision'),
                'rationale': inspected_payload.get('decision_rationale'),
                'failure_tag': inspected_payload.get('failure_tag'),
            }
            print_json('inspected_memory_entry', structured_memory_view)
            """
        )
    )

    cells.append(
        markdown(
            """
            ## 10. Final State Verification

            This cell first proves the pending-run guard at the state-machine level, then prints the final persisted artifacts: `results.tsv`, the memory ledger, the final best metric, and the total experiment count. It also stops the API server cleanly after verification. No further input is needed.
            """
        )
    )

    cells.append(
        code(
            """
            pending_guard_packet = build_synthetic_packet(val_bpb=0.970000)
            pending_guard_packet['description'] = 'Pending guard probe'
            pending_guard_packet['objective'] = (
                'Modify train.py by adding a smoke-test comment so the control plane can prove '
                'it rejects a second run while a candidate is still pending review.'
            )
            first_guard_status, first_guard_result = http_post_json(
                f'{API_BASE}/run',
                {'packet': pending_guard_packet},
            )
            assert first_guard_status == 200, first_guard_result
            second_guard_status, second_guard_result = http_post_json(
                f'{API_BASE}/run',
                {'packet': pending_guard_packet},
            )
            assert second_guard_status == 409, second_guard_result
            cleanup_status, cleanup_result = http_post_json(
                f'{API_BASE}/discard',
                {'rationale': 'cleanup after pending guard probe'},
            )
            assert cleanup_status == 200, cleanup_result

            results_text = (NODE_ROOT / 'results.tsv').read_text(encoding='utf-8')
            memory_text = (NODE_ROOT / 'experiment_memory.jsonl').read_text(encoding='utf-8')
            final_status = http_get_json(f'{API_BASE}/status')
            total_experiment_count = max(len(results_text.splitlines()) - 1, 0)

            print_json(
                'pending_guard_probe',
                {
                    'first_run_status': first_guard_status,
                    'second_run_status': second_guard_status,
                    'second_run_payload': second_guard_result,
                    'cleanup_decision': cleanup_result,
                },
            )
            print('results.tsv:')
            print(results_text)
            print('memory file:')
            print(memory_text)
            print('final best metric:', final_status['state']['best_bpb'])
            print('total experiment count:', total_experiment_count)

            stop_api_server()
            print(f'API server stopped. Artifacts remain at {NODE_ROOT} until the notebook kernel exits.')
            """
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "notebooks"
    output = root / "stage1_sudhir_demo_verification_v2.ipynb"
    output.write_text(json.dumps(build_notebook(), indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
