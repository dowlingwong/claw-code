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
from src.autoresearch_worker import (
    AutoresearchExperimentPacket,
    autoresearch_status,
    discard_autoresearch_candidate,
    ensure_autoresearch_baseline,
    ensure_autoresearch_branch,
    keep_autoresearch_candidate,
    load_autoresearch_packet,
    load_autoresearch_state,
    loop_autoresearch,
    run_autoresearch_packet,
)
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

            result = keep_autoresearch_candidate(root)
            state = load_autoresearch_state(root)
            rows = (root / 'results.tsv').read_text(encoding='utf-8').splitlines()
            self.assertEqual(result['decision'], 'keep')
            self.assertIsNone(state.pending_experiment)
            self.assertEqual(state.last_decision, 'keep')
            self.assertEqual(state.best_bpb, 0.98)
            self.assertEqual(len(rows), 2)
            self.assertIn('\tkeep\tKeep candidate', rows[1])

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

            discard = discard_autoresearch_candidate(root)
            self.assertEqual(discard['decision'], 'discard')
            self.assertEqual(short_head_commit(root), discard['reverted_to_commit'])
            self.assertIn('candidate', (root / 'train.py').read_text(encoding='utf-8'))
            self.assertNotEqual(short_head_commit(root), 'unknown')

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
