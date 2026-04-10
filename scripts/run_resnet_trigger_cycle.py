from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_NODE_ROOT = Path(__file__).resolve().parents[3] / "nodes" / "ResNet_trigger"
CLAW_ROOT = Path(__file__).resolve().parents[1]


def pick_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_get_json(url: str, timeout: int = 30) -> dict:
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_post_json(url: str, payload: dict, timeout: int = 300) -> tuple[int, dict]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace").strip()
        parsed = json.loads(body) if body else {"error": str(error)}
        return error.code, parsed


def ensure_node_shape(node_root: Path) -> None:
    for name in ("prepare.py", "train.py", "program.md", "pyproject.toml"):
        if not (node_root / name).exists():
            raise FileNotFoundError(f"Expected node file is missing: {node_root / name}")
    if not (node_root / "resnet_1d.py").exists():
        raise FileNotFoundError(f"Expected backbone module is missing: {node_root / 'resnet_1d.py'}")
    if not shutil_which("uv"):
        raise RuntimeError("`uv` is not installed or not on PATH.")


def shutil_which(name: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def start_api_server(
    node_root: Path,
    port: int,
    backend_kind: str,
    model: str,
    host: str,
    device: str,
) -> tuple[subprocess.Popen[str], str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CLAW_ROOT)
    env["RESNET_TRIGGER_DEVICE"] = device
    command = [
        sys.executable,
        "-m",
        "src.main",
        "api-server",
        "--root",
        str(node_root),
        "--port",
        str(port),
        "--listen",
        "127.0.0.1",
        "--backend",
        backend_kind,
        "--model",
        model,
        "--host",
        host,
    ]
    proc = subprocess.Popen(
        command,
        cwd=CLAW_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    api_base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 15
    last_error = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout = proc.stdout.read() if proc.stdout else ""
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"API server exited early.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        try:
            payload = http_get_json(f"{api_base}/health", timeout=2)
            if payload.get("status") == "ok":
                return proc, api_base
        except Exception as error:  # noqa: BLE001
            last_error = str(error)
            time.sleep(0.25)
    raise RuntimeError(f"API server did not become healthy: {last_error}")


def stop_api_server(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def read_best_performance(node_root: Path) -> dict:
    path = node_root / "artifacts" / "best_performance.json"
    return json.loads(path.read_text(encoding="utf-8"))


def verify_artifacts(node_root: Path) -> dict:
    artifacts_dir = node_root / "artifacts"
    best_model = artifacts_dir / "best_model.pt"
    best_metrics = artifacts_dir / "best_performance.json"
    metrics_latest = artifacts_dir / "metrics_latest.json"
    history_latest = artifacts_dir / "history_latest.json"
    timing_latest = artifacts_dir / "timing_latest.json"
    required = [best_model, best_metrics, metrics_latest, history_latest, timing_latest]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Expected artifact(s) missing: {missing}")
    best_payload = json.loads(best_metrics.read_text(encoding="utf-8"))
    return {
        "artifacts_dir": str(artifacts_dir),
        "best_model_path": str(best_model),
        "best_model_size_bytes": best_model.stat().st_size,
        "best_performance_path": str(best_metrics),
        "best_performance": best_payload,
        "metrics_latest_path": str(metrics_latest),
        "history_latest_path": str(history_latest),
        "timing_latest_path": str(timing_latest),
    }


def build_packet(learning_rate: float, timeout_seconds: int) -> dict:
    return {
        "objective": (
            f"Modify train.py by setting LEARNING_RATE = {learning_rate:.6f} "
            "to test whether validation AUC improves while keeping the run stable."
        ),
        "description": f"Set LEARNING_RATE = {learning_rate:.6f} for control-plane verification",
        "train_command": "uv run train.py > run.log 2>&1",
        "timeout_seconds": timeout_seconds,
        "log_path": "run.log",
        "results_tsv": "results.tsv",
        "syntax_check_command": "python3 -m py_compile train.py",
    }


def build_rationale(candidate_auc: float, baseline_auc: float, keep: bool) -> str:
    delta = candidate_auc - baseline_auc
    if keep:
        return (
            f"Keeping candidate because best validation AUC improved by {delta:.6f} "
            f"(baseline {baseline_auc:.6f} -> candidate {candidate_auc:.6f})."
        )
    return (
        f"Discarding candidate because best validation AUC did not improve "
        f"(baseline {baseline_auc:.6f} -> candidate {candidate_auc:.6f}, delta {delta:.6f})."
    )


def unique_branch_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"autoresearch/resnet-trigger-demo-{stamp}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one full control-plane cycle against the ResNet_trigger node and verify best-model artifacts."
    )
    parser.add_argument("--node-root", default=str(DEFAULT_NODE_ROOT))
    parser.add_argument("--backend", default="ollama", choices=["ollama", "openai-compat"])
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--port", type=int, default=0, help="API port; defaults to a free local port")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--candidate-learning-rate", type=float, default=5e-4)
    parser.add_argument("--device", default=os.environ.get("RESNET_TRIGGER_DEVICE", "mps"))
    args = parser.parse_args(argv)

    node_root = Path(args.node_root).resolve()
    ensure_node_shape(node_root)
    api_proc: subprocess.Popen[str] | None = None
    api_base = ""

    try:
        port = args.port or pick_free_port()
        api_proc, api_base = start_api_server(
            node_root=node_root,
            port=port,
            backend_kind=args.backend,
            model=args.model,
            host=args.host,
            device=args.device,
        )

        branch_name = unique_branch_name()
        isolate_status, isolate_result = http_post_json(
            f"{api_base}/isolate",
            {"branch": branch_name, "create": True},
        )
        if isolate_status != 200:
            raise RuntimeError(f"Failed to isolate node branch: {isolate_result}")

        baseline_status, baseline_result = http_post_json(
            f"{api_base}/baseline",
            {
                "command": "uv run train.py > run.log 2>&1",
                "log_path": "run.log",
                "timeout_seconds": args.timeout_seconds,
                "results_tsv": "results.tsv",
            },
            timeout=max(args.timeout_seconds + 60, 300),
        )
        if baseline_status != 200:
            raise RuntimeError(f"Baseline failed: {baseline_result}")

        baseline_artifacts = verify_artifacts(node_root)
        baseline_performance = read_best_performance(node_root)
        baseline_auc = float(baseline_performance["best_val_auc"])

        run_status, run_result = http_post_json(
            f"{api_base}/run",
            {
                "packet": build_packet(args.candidate_learning_rate, args.timeout_seconds),
                "trace": True,
            },
            timeout=max(args.timeout_seconds + 120, 300),
        )
        if run_status != 200:
            raise RuntimeError(f"Candidate run failed: {run_result}")

        candidate_artifacts = verify_artifacts(node_root)
        candidate_performance = read_best_performance(node_root)
        candidate_auc = float(candidate_performance["best_val_auc"])
        recommended_status = str(run_result.get("recommended_status", "discard"))
        keep_candidate = recommended_status == "keep"
        rationale = build_rationale(candidate_auc, baseline_auc, keep_candidate)
        decision_endpoint = "keep" if keep_candidate else "discard"
        decision_status, decision_result = http_post_json(
            f"{api_base}/{decision_endpoint}",
            {"rationale": rationale},
        )
        if decision_status != 200:
            raise RuntimeError(f"Decision failed: {decision_result}")

        final_status = http_get_json(f"{api_base}/status")
        memory_latest = http_get_json(f"{api_base}/memory?limit=2")
        results_tsv = (node_root / "results.tsv").read_text(encoding="utf-8")

        summary = {
            "node_root": str(node_root),
            "api_base": api_base,
            "device": args.device,
            "branch": branch_name,
            "baseline_result": baseline_result,
            "baseline_best_performance": baseline_performance,
            "baseline_artifacts": baseline_artifacts,
            "candidate_packet": build_packet(args.candidate_learning_rate, args.timeout_seconds),
            "run_result": run_result,
            "candidate_best_performance": candidate_performance,
            "candidate_artifacts": candidate_artifacts,
            "decision_endpoint": decision_endpoint,
            "decision_rationale": rationale,
            "decision_result": decision_result,
            "final_status": final_status,
            "latest_memory": memory_latest,
            "results_tsv": results_tsv,
        }
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        stop_api_server(api_proc)


if __name__ == "__main__":
    raise SystemExit(main())
