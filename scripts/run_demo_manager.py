from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.manager_prompt import MANAGER_SYSTEM_PROMPT


def http_get_json(url: str) -> dict:
    with urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode('utf-8'))


def http_post_json(url: str, payload: dict) -> tuple[int, dict]:
    request = Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read().decode('utf-8'))
    except HTTPError as error:
        body = error.read().decode('utf-8', errors='replace').strip()
        try:
            payload = json.loads(body) if body else {'error': str(error)}
        except json.JSONDecodeError:
            payload = {'error': body or str(error)}
        return error.code, payload


def build_rationale(run_result: dict) -> str:
    recommended = str(run_result.get('recommended_status', 'discard'))
    experiment = run_result.get('experiment', {})
    if not isinstance(experiment, dict):
        experiment = {}
    val_bpb = experiment.get('val_bpb')
    if recommended == 'keep':
        return f'val_bpb improved to {val_bpb}'
    if recommended == 'crash':
        return f'run failed: {experiment.get("error") or "invalid experiment result"}'
    return f'val_bpb did not improve; observed {val_bpb}'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run one stage-one manager iteration against the autoresearch API.')
    parser.add_argument('--api-base', default='http://127.0.0.1:7331')
    parser.add_argument('--packet-path', required=True)
    args = parser.parse_args(argv)

    packet_path = Path(args.packet_path).resolve()
    if not packet_path.exists():
        print(f'packet not found: {packet_path}', file=sys.stderr)
        return 1

    api_base = args.api_base.rstrip('/')

    try:
        status_before = http_get_json(f'{api_base}/status')
        memory_before = http_get_json(f'{api_base}/memory?limit=5')
        memory_summary_before = http_get_json(f'{api_base}/memory-summary?limit=5')
    except (URLError, json.JSONDecodeError) as error:
        print(f'failed to read initial API state: {error}', file=sys.stderr)
        return 1

    state_before = status_before.get('state', {})
    if isinstance(state_before, dict) and state_before.get('pending_experiment') is not None:
        print(json.dumps({
            'manager_prompt_loaded': len(MANAGER_SYSTEM_PROMPT.split()),
            'status': 'blocked',
            'reason': 'pending experiment already exists',
            'state': status_before,
        }, indent=2))
        return 2

    run_status, run_result = http_post_json(
        f'{api_base}/run',
        {'packet_path': str(packet_path)},
    )
    if run_status != 200:
        print(json.dumps({
            'manager_prompt_loaded': len(MANAGER_SYSTEM_PROMPT.split()),
            'status': 'failed',
            'run_status': run_status,
            'run_result': run_result,
        }, indent=2))
        return 1

    recommended = str(run_result.get('recommended_status', 'discard'))
    rationale = build_rationale(run_result)
    decision_endpoint = 'keep' if recommended == 'keep' else 'discard'
    decision_status, decision_result = http_post_json(
        f'{api_base}/{decision_endpoint}',
        {'rationale': rationale},
    )
    if decision_status != 200:
        print(json.dumps({
            'manager_prompt_loaded': len(MANAGER_SYSTEM_PROMPT.split()),
            'status': 'failed',
            'decision_status': decision_status,
            'decision_result': decision_result,
        }, indent=2))
        return 1

    memory_after = http_get_json(f'{api_base}/memory?limit=1')
    memory_summary_after = http_get_json(f'{api_base}/memory-summary?limit=5')
    summary = {
        'manager_prompt_loaded': len(MANAGER_SYSTEM_PROMPT.split()),
        'status_before': status_before,
        'memory_before_count': len(memory_before.get('memory', [])),
        'memory_summary_before': memory_summary_before,
        'run_result': run_result,
        'decision': decision_result,
        'memory_after_latest': (memory_after.get('memory') or [None])[-1],
        'memory_summary_after': memory_summary_after,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
