from __future__ import annotations

import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.autoresearch_runner import (
    ExperimentMetrics,
    append_results_row,
    best_recorded_bpb,
    parse_run_log,
    short_head_commit,
    setup_autoresearch,
)
from src.agent_loop import infer_edit_tool_call_from_prompt
from src.autoresearch_worker import (
    AutoresearchExperimentPacket,
    autoresearch_status,
    discard_autoresearch_candidate,
    ensure_autoresearch_baseline,
    ensure_autoresearch_branch,
    keep_autoresearch_candidate,
    load_memory_events,
    load_autoresearch_packet,
    load_autoresearch_state,
    loop_autoresearch,
    run_autoresearch_packet,
    summarize_memory_events,
)
from src.demo_verification import build_diff_summary, extract_metric_from_text, get_changed_files, snapshot_workspace_sources
from src.worker_api import WorkerRecord


class AutoresearchIntegrationTests(unittest.TestCase):
    def init_repo(self, root: Path) -> None:
        subprocess.run(['git', 'init'], cwd=root, check=True, capture_output=True, text=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=root, check=True, capture_output=True, text=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=root, check=True, capture_output=True, text=True)
        subprocess.run(['git', 'add', '.'], cwd=root, check=True, capture_output=True, text=True)
        subprocess.run(['git', 'commit', '-m', 'init'], cwd=root, check=True, capture_output=True, text=True)

    def test_setup_autoresearch_initializes_results(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ('prepare.py', 'train.py', 'program.md'):
                (root / name).write_text('# stub\n', encoding='utf-8')
            self.init_repo(root)

            report = setup_autoresearch(root)
            self.assertTrue((root / 'results.tsv').exists())
            self.assertEqual(Path(report.root).resolve(), root.resolve())
            self.assertIn('results.tsv', report.results_tsv_path)

    def test_parse_run_log_extracts_metrics(self) -> None:
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'run.log'
            log_path.write_text(
                '\n'.join(
                    [
                        '---',
                        'val_bpb:          0.997900',
                        'training_seconds: 300.1',
                        'total_seconds:    325.9',
                        'peak_vram_mb:     45060.2',
                        'mfu_percent:      39.80',
                        'total_tokens_M:   499.6',
                        'num_steps:        953',
                        'num_params_M:     50.3',
                        'depth:            8',
                    ]
                ),
                encoding='utf-8',
            )
            metrics = parse_run_log(log_path)
            self.assertTrue(metrics.success)
            self.assertEqual(metrics.val_bpb, 0.9979)
            self.assertEqual(metrics.depth, 8)

    def test_extract_metric_from_text_accepts_equals_and_scientific_notation(self) -> None:
        text = '\n'.join(
            [
                'candidate metric follows',
                'val_bpb = 9.979e-01',
                'depth: 8',
            ]
        )
        self.assertEqual(extract_metric_from_text(text), 0.9979)

    def test_append_results_row_and_best_bpb(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ('prepare.py', 'train.py', 'program.md'):
                (root / name).write_text('# stub\n', encoding='utf-8')
            metrics = ExperimentMetrics(
                success=True,
                log_path=str(root / 'run.log'),
                timed_out=False,
                return_code=0,
                val_bpb=0.99,
                peak_vram_mb=1024.0,
            )
            append_results_row(root, 'abc1234', metrics, 'keep', 'baseline')
            append_results_row(root, 'def5678', metrics, 'discard', 'worse')
            self.assertEqual(best_recorded_bpb(root), 0.99)

    def test_load_autoresearch_packet(self) -> None:
        with TemporaryDirectory() as tmpdir:
            packet_path = Path(tmpdir) / 'experiment.json'
            packet_path.write_text(
                json.dumps(
                    {
                        'objective': 'Try one change',
                        'description': 'Small optimizer tweak',
                    }
                ),
                encoding='utf-8',
            )
            packet = load_autoresearch_packet(packet_path)
            self.assertEqual(packet.objective, 'Try one change')
            self.assertEqual(packet.timeout_seconds, 600)

    def test_run_autoresearch_packet_wraps_worker_and_runner(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ('prepare.py', 'program.md'):
                (root / name).write_text('# stub\n', encoding='utf-8')
            (root / 'train.py').write_text('print("hello")\n', encoding='utf-8')
            self.init_repo(root)

            packet = AutoresearchExperimentPacket(
                objective='Try a small change',
                description='Tweak train.py',
            )
            worker = WorkerRecord(
                worker_id='worker123',
                root=str(root),
                model='qwen2.5-coder:7b',
                host='http://localhost:11434',
                state='finished',
                created_at='2026-04-07T00:00:00Z',
                updated_at='2026-04-07T00:00:01Z',
                run_count=1,
                last_packet={},
                last_result={
                    'changed_files': ['train.py'],
                    'tool_calls': ['edit_file'],
                    'tool_trace': [],
                    'verification': {'acceptance_tests': []},
                },
            )
            (root / 'train.py').write_text('print("changed")\n', encoding='utf-8')

            with patch('src.autoresearch_worker.create_worker', return_value=worker):
                with patch('src.autoresearch_worker.run_worker', return_value=worker):
                    with patch(
                        'src.autoresearch_worker.run_experiment',
                        return_value=ExperimentMetrics(
                            success=True,
                            log_path=str(root / 'run.log'),
                            timed_out=False,
                            return_code=0,
                            val_bpb=0.98,
                            peak_vram_mb=2048.0,
                        ),
                    ):
                        result = run_autoresearch_packet(packet, root=root, trace=False)

            self.assertEqual(result['recommended_status'], 'keep')
            self.assertEqual(result['experiment']['val_bpb'], 0.98)
            self.assertTrue((root / 'results.tsv').exists())
            self.assertIsNotNone(result['state']['pending_experiment'])
            self.assertEqual((root / 'results.tsv').read_text(encoding='utf-8').count('\n'), 1)
            memory = load_memory_events(root)
            self.assertTrue(memory)
            candidate = memory[-1]['payload']
            self.assertEqual(candidate['experiment_id'], result['commit'])
            self.assertEqual(candidate['code_change_summary'], ['train.py'])
            self.assertEqual(candidate['decision_rationale'], '')
            self.assertEqual(candidate['failure_tag'], '')
            self.assertEqual(candidate['idea_key'], 'tweak-train-py')
            summary = summarize_memory_events(root, limit=5)
            self.assertEqual(summary['memory_event_count'], len(memory))
            self.assertEqual(summary['idea_rollup'][0]['idea_key'], 'tweak-train-py')
            self.assertEqual(summary['recent_events'][-1]['event'], 'candidate_run')

    def test_run_autoresearch_packet_applies_deterministic_assignment_fallback_after_no_change(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ('prepare.py', 'program.md'):
                (root / name).write_text('# stub\n', encoding='utf-8')
            (root / 'train.py').write_text('LEARNING_RATE = 1e-3\nprint("hello")\n', encoding='utf-8')
            self.init_repo(root)

            packet = AutoresearchExperimentPacket(
                objective='Modify train.py by setting LEARNING_RATE = 0.0005 to test a smaller step size.',
                description='Set LEARNING_RATE = 0.0005',
            )
            worker = WorkerRecord(
                worker_id='worker123',
                root=str(root),
                model='qwen2.5-coder:7b',
                host='http://localhost:11434',
                state='failed',
                created_at='2026-04-07T00:00:00Z',
                updated_at='2026-04-07T00:00:01Z',
                run_count=1,
                last_packet={},
                last_result={
                    'changed_files': [],
                    'tool_calls': [],
                    'tool_trace': [],
                    'verification': {'acceptance_tests': []},
                    'final_answer': '',
                    'stop_reason': 'error',
                },
                last_error='Agent stopped after reaching max_turns=8.',
            )

            with patch('src.autoresearch_worker.create_worker', return_value=worker):
                with patch('src.autoresearch_worker.run_worker', return_value=worker):
                    with patch(
                        'src.autoresearch_worker.run_experiment',
                        return_value=ExperimentMetrics(
                            success=True,
                            log_path=str(root / 'run.log'),
                            timed_out=False,
                            return_code=0,
                            val_bpb=0.97,
                            peak_vram_mb=1024.0,
                        ),
                    ):
                        result = run_autoresearch_packet(packet, root=root, trace=False)

            self.assertEqual(result['recommended_status'], 'keep')
            self.assertIn('LEARNING_RATE = 0.0005', (root / 'train.py').read_text(encoding='utf-8'))
            self.assertEqual(result['worker']['last_result']['changed_files'], ['train.py'])
            self.assertEqual(result['worker']['state'], 'finished')

    def test_run_autoresearch_packet_repairs_invalid_assignment_syntax(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ('prepare.py', 'program.md'):
                (root / name).write_text('# stub\n', encoding='utf-8')
            (root / 'train.py').write_text('DROPOUT = 0.0\nprint("hello")\n', encoding='utf-8')
            self.init_repo(root)
            (root / 'train.py').write_text('DROPOUT = 0.100000.\nprint("hello")\n', encoding='utf-8')

            packet = AutoresearchExperimentPacket(
                objective='Modify train.py by setting DROPOUT = 0.100000 for stronger regularization.',
                description='Set DROPOUT = 0.100000',
            )
            worker = WorkerRecord(
                worker_id='worker123',
                root=str(root),
                model='qwen2.5-coder:7b',
                host='http://localhost:11434',
                state='failed',
                created_at='2026-04-07T00:00:00Z',
                updated_at='2026-04-07T00:00:01Z',
                run_count=1,
                last_packet={},
                last_result={
                    'changed_files': ['train.py'],
                    'tool_calls': ['edit_file'],
                    'tool_trace': [],
                    'verification': {
                        'acceptance_tests': [
                            {
                                'command': 'python3 -m py_compile train.py',
                                'success': False,
                                'exit_code': 1,
                                'output': 'SyntaxError',
                            }
                        ]
                    },
                    'final_answer': '',
                    'stop_reason': 'verification_failed',
                },
                last_error='SyntaxError',
            )

            with patch('src.autoresearch_worker.create_worker', return_value=worker):
                with patch('src.autoresearch_worker.run_worker', return_value=worker):
                    with patch(
                        'src.autoresearch_worker.run_experiment',
                        return_value=ExperimentMetrics(
                            success=True,
                            log_path=str(root / 'run.log'),
                            timed_out=False,
                            return_code=0,
                            val_bpb=0.96,
                            peak_vram_mb=1024.0,
                        ),
                    ):
                        result = run_autoresearch_packet(packet, root=root, trace=False)

            self.assertEqual(result['recommended_status'], 'keep')
            self.assertIn('DROPOUT = 0.100000', (root / 'train.py').read_text(encoding='utf-8'))
            self.assertNotIn('0.100000.', (root / 'train.py').read_text(encoding='utf-8'))
            self.assertEqual(result['worker']['state'], 'finished')

    def test_keep_autoresearch_candidate_records_keep(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ('prepare.py', 'program.md'):
                (root / name).write_text('# stub\n', encoding='utf-8')
            (root / 'train.py').write_text('print("hello")\n', encoding='utf-8')
            self.init_repo(root)

            packet = AutoresearchExperimentPacket(objective='Try a small change', description='Keep candidate')
            worker = WorkerRecord(
                worker_id='worker123',
                root=str(root),
                model='qwen2.5-coder:7b',
                host='http://localhost:11434',
                state='finished',
                created_at='2026-04-07T00:00:00Z',
                updated_at='2026-04-07T00:00:01Z',
                run_count=1,
                last_packet={},
                last_result={
                    'changed_files': ['train.py'],
                    'tool_calls': ['edit_file'],
                    'tool_trace': [],
                    'verification': {'acceptance_tests': []},
                },
            )
            (root / 'train.py').write_text('print("changed")\n', encoding='utf-8')

            with patch('src.autoresearch_worker.create_worker', return_value=worker):
                with patch('src.autoresearch_worker.run_worker', return_value=worker):
                    with patch(
                        'src.autoresearch_worker.run_experiment',
                        return_value=ExperimentMetrics(
                            success=True,
                            log_path=str(root / 'run.log'),
                            timed_out=False,
                            return_code=0,
                            val_bpb=0.98,
                            peak_vram_mb=2048.0,
                        ),
                    ):
                        run_autoresearch_packet(packet, root=root, trace=False)

            result = keep_autoresearch_candidate(root, rationale='val_bpb improved 0.03 vs baseline')
            state = load_autoresearch_state(root)
            rows = (root / 'results.tsv').read_text(encoding='utf-8').splitlines()
            self.assertEqual(result['decision'], 'keep')
            self.assertIsNone(state.pending_experiment)
            self.assertEqual(state.last_decision, 'keep')
            self.assertEqual(state.best_bpb, 0.98)
            self.assertEqual(len(rows), 2)
            self.assertIn('\tkeep\tKeep candidate', rows[1])
            memory = load_memory_events(root)
            latest = memory[-1]['payload']
            self.assertEqual(latest['decision_rationale'], 'val_bpb improved 0.03 vs baseline')
            self.assertEqual(latest['code_change_summary'], ['train.py'])
            self.assertEqual(latest['failure_tag'], '')
            self.assertEqual(latest['idea_key'], 'keep-candidate')
            summary = summarize_memory_events(root, limit=5)
            self.assertEqual(summary['latest_decision']['event'], 'keep')
            self.assertEqual(summary['idea_rollup'][0]['keeps'], 1)

    def test_discard_autoresearch_candidate_resets_repo(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ('prepare.py', 'program.md'):
                (root / name).write_text('# stub\n', encoding='utf-8')
            (root / 'train.py').write_text('print("hello")\n', encoding='utf-8')
            self.init_repo(root)
            packet = AutoresearchExperimentPacket(objective='Try a small change', description='Discard candidate')
            worker = WorkerRecord(
                worker_id='worker123',
                root=str(root),
                model='qwen2.5-coder:7b',
                host='http://localhost:11434',
                state='finished',
                created_at='2026-04-07T00:00:00Z',
                updated_at='2026-04-07T00:00:01Z',
                run_count=1,
                last_packet={},
                last_result={
                    'changed_files': ['train.py'],
                    'tool_calls': ['edit_file'],
                    'tool_trace': [],
                    'verification': {'acceptance_tests': []},
                },
            )
            (root / 'train.py').write_text('print("candidate")\n', encoding='utf-8')

            with patch('src.autoresearch_worker.create_worker', return_value=worker):
                with patch('src.autoresearch_worker.run_worker', return_value=worker):
                    with patch(
                        'src.autoresearch_worker.run_experiment',
                        return_value=ExperimentMetrics(
                            success=True,
                            log_path=str(root / 'run.log'),
                            timed_out=False,
                            return_code=0,
                            val_bpb=1.2,
                            peak_vram_mb=2048.0,
                        ),
                    ):
                        result = run_autoresearch_packet(packet, root=root, trace=False)

            self.assertEqual(result['recommended_status'], 'keep')
            keep_autoresearch_candidate(root)
            (root / 'train.py').write_text('print("worse")\n', encoding='utf-8')
            with patch('src.autoresearch_worker.create_worker', return_value=worker):
                with patch('src.autoresearch_worker.run_worker', return_value=worker):
                    with patch(
                        'src.autoresearch_worker.run_experiment',
                        return_value=ExperimentMetrics(
                            success=True,
                            log_path=str(root / 'run.log'),
                            timed_out=False,
                            return_code=0,
                            val_bpb=1.3,
                            peak_vram_mb=4096.0,
                        ),
                    ):
                        run_autoresearch_packet(packet, root=root, trace=False)

            discard = discard_autoresearch_candidate(root, rationale='no improvement over current best')
            self.assertEqual(discard['decision'], 'discard')
            self.assertEqual(short_head_commit(root), discard['reverted_to_commit'])
            self.assertIn('candidate', (root / 'train.py').read_text(encoding='utf-8'))
            self.assertNotEqual(short_head_commit(root), 'unknown')
            memory = load_memory_events(root)
            latest = memory[-1]['payload']
            self.assertEqual(latest['decision_rationale'], 'no improvement over current best')
            self.assertEqual(latest['code_change_summary'], ['train.py'])
            self.assertEqual(latest['failure_tag'], 'no_improvement')
            summary = summarize_memory_events(root, limit=5)
            self.assertEqual(summary['latest_decision']['event'], 'discard')
            self.assertTrue(summary['idea_rollup'])

    def test_ensure_autoresearch_baseline_records_first_run(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ('prepare.py', 'program.md'):
                (root / name).write_text('# stub\n', encoding='utf-8')
            (root / 'train.py').write_text('print("hello")\n', encoding='utf-8')
            self.init_repo(root)

            with patch(
                'src.autoresearch_worker.run_experiment',
                return_value=ExperimentMetrics(
                    success=True,
                    log_path=str(root / 'run.log'),
                    timed_out=False,
                    return_code=0,
                    val_bpb=1.01,
                    peak_vram_mb=1024.0,
                ),
            ):
                baseline = ensure_autoresearch_baseline(root)

            state = load_autoresearch_state(root)
            self.assertTrue(baseline['baseline_created'])
            self.assertEqual(state.baseline_bpb, 1.01)
            self.assertEqual(state.best_bpb, 1.01)
            self.assertIn('\tkeep\tbaseline', (root / 'results.tsv').read_text(encoding='utf-8'))

    def test_ensure_autoresearch_branch_creates_isolated_branch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ('prepare.py', 'program.md'):
                (root / name).write_text('# stub\n', encoding='utf-8')
            (root / 'train.py').write_text('print("hello")\n', encoding='utf-8')
            self.init_repo(root)

            result = ensure_autoresearch_branch(root, branch='autoresearch/test', create=True)
            status = autoresearch_status(root)
            self.assertEqual(result['current_branch'], 'autoresearch/test')
            self.assertTrue(status['isolated_branch'])
            self.assertEqual(status['current_branch'], 'autoresearch/test')

    def test_loop_autoresearch_runs_keep_cycle(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ('prepare.py', 'program.md'):
                (root / name).write_text('# stub\n', encoding='utf-8')
            (root / 'train.py').write_text('print("hello")\n', encoding='utf-8')
            self.init_repo(root)
            ensure_autoresearch_branch(root, branch='autoresearch/test', create=True)

            packet = AutoresearchExperimentPacket(objective='Loop once', description='Loop candidate')
            worker = WorkerRecord(
                worker_id='worker123',
                root=str(root),
                model='qwen2.5-coder:7b',
                host='http://localhost:11434',
                state='finished',
                created_at='2026-04-07T00:00:00Z',
                updated_at='2026-04-07T00:00:01Z',
                run_count=1,
                last_packet={},
                last_result={
                    'changed_files': ['train.py'],
                    'tool_calls': ['edit_file'],
                    'tool_trace': [],
                    'verification': {'acceptance_tests': []},
                },
            )
            (root / 'train.py').write_text('print("changed")\n', encoding='utf-8')

            with patch(
                'src.autoresearch_worker.ensure_autoresearch_baseline',
                return_value={'baseline_created': False, 'reason': 'baseline already recorded'},
            ):
                with patch('src.autoresearch_worker.create_worker', return_value=worker):
                    with patch('src.autoresearch_worker.run_worker', return_value=worker):
                        with patch(
                            'src.autoresearch_worker.run_experiment',
                            return_value=ExperimentMetrics(
                                success=True,
                                log_path=str(root / 'run.log'),
                                timed_out=False,
                                return_code=0,
                                val_bpb=0.97,
                                peak_vram_mb=2048.0,
                            ),
                        ):
                            result = loop_autoresearch(packet, root=root, iterations=1, retry_limit=0)

            self.assertEqual(len(result['history']), 1)
            self.assertEqual(result['history'][0]['decision']['decision'], 'keep')

    def test_memory_reader_and_pending_run_guard_inputs_are_available(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ('prepare.py', 'program.md'):
                (root / name).write_text('# stub\n', encoding='utf-8')
            (root / 'train.py').write_text('print("hello")\n', encoding='utf-8')
            self.init_repo(root)

            packet = AutoresearchExperimentPacket(
                objective='Try a small change',
                description='Tweak train.py',
            )
            worker = WorkerRecord(
                worker_id='worker123',
                root=str(root),
                model='qwen2.5-coder:7b',
                host='http://localhost:11434',
                state='finished',
                created_at='2026-04-07T00:00:00Z',
                updated_at='2026-04-07T00:00:01Z',
                run_count=1,
                last_packet={},
                last_result={
                    'changed_files': ['train.py'],
                    'tool_calls': ['edit_file'],
                    'tool_trace': [],
                    'verification': {'acceptance_tests': []},
                },
            )
            (root / 'train.py').write_text('print("changed")\n', encoding='utf-8')
            with patch('src.autoresearch_worker.create_worker', return_value=worker):
                with patch('src.autoresearch_worker.run_worker', return_value=worker):
                    with patch(
                        'src.autoresearch_worker.run_experiment',
                        return_value=ExperimentMetrics(
                            success=True,
                            log_path=str(root / 'run.log'),
                            timed_out=False,
                            return_code=0,
                            val_bpb=0.98,
                            peak_vram_mb=2048.0,
                        ),
                    ):
                        run_result = run_autoresearch_packet(packet, root=root, trace=False)

            memory = load_memory_events(root, limit=1)
            self.assertEqual(len(memory), 1)
            self.assertEqual(memory[0]['payload']['experiment_id'], run_result['commit'])

            state = load_autoresearch_state(root)
            self.assertIsNotNone(state.pending_experiment)

    def test_demo_verification_snapshot_and_diff_summary(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / 'train.py').write_text('DEMO_OFFSET = 0.0\nprint("hi")\n', encoding='utf-8')
            (root / 'prepare.py').write_text('# stub\n', encoding='utf-8')
            (root / 'program.md').write_text('demo\n', encoding='utf-8')

            before = snapshot_workspace_sources(root)
            (root / 'train.py').write_text('DEMO_OFFSET = -0.03\nprint("hi")\n', encoding='utf-8')
            after = snapshot_workspace_sources(root)

            self.assertEqual(get_changed_files(before, after), ['train.py'])
            summary = build_diff_summary(before, after)
            self.assertEqual(summary[0]['path'], 'train.py')
            self.assertGreater(summary[0]['changed_line_count'], 0)
            self.assertIn('-DEMO_OFFSET = 0.0', summary[0]['diff'])
            self.assertIn('+DEMO_OFFSET = -0.03', summary[0]['diff'])

    def test_infer_edit_tool_call_from_prompt_supports_assignment_replacement(self) -> None:
        tool_call = infer_edit_tool_call_from_prompt(
            'Modify train.py by setting DEMO_OFFSET = -0.030 so the deterministic validation metric improves.'
        )
        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call['name'], 'run_shell_command')
        self.assertIn('DEMO_OFFSET', tool_call['arguments']['command'])
        self.assertIn('-0.030', tool_call['arguments']['command'])

    def test_autoresearch_cli_setup_and_parse_log(self) -> None:
        from src.main import main

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ('prepare.py', 'train.py', 'program.md'):
                (root / name).write_text('# stub\n', encoding='utf-8')
            self.init_repo(root)
            log_path = root / 'run.log'
            log_path.write_text('val_bpb:          1.000000\n', encoding='utf-8')

            with patch('sys.stdout', new=io.StringIO()):
                setup_exit = main(['autoresearch', 'setup', '--root', str(root)])
            with patch('sys.stdout', new=io.StringIO()):
                status_exit = main(['autoresearch', 'status', '--root', str(root)])
            with patch('sys.stdout', new=io.StringIO()):
                parse_exit = main(['autoresearch', 'parse-log', '--root', str(root), 'run.log'])

            self.assertEqual(setup_exit, 0)
            self.assertEqual(status_exit, 0)
            self.assertEqual(parse_exit, 0)


if __name__ == '__main__':
    unittest.main()
