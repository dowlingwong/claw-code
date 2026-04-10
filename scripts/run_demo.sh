#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AUTORESEARCH_ROOT="${AUTORESEARCH_ROOT:-${REPO_ROOT}/../../nodes/autoresearch-macos}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-17331}"
DEMO_PACKET="${DEMO_PACKET:-${REPO_ROOT}/notebooks/autoresearch_smoke_packet.json}"
API_BASE="http://${API_HOST}:${API_PORT}"

if [[ ! -d "${AUTORESEARCH_ROOT}" ]]; then
  echo "AUTORESEARCH_ROOT does not exist: ${AUTORESEARCH_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${DEMO_PACKET}" ]]; then
  echo "DEMO_PACKET does not exist: ${DEMO_PACKET}" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${API_PID:-}" ]] && kill -0 "${API_PID}" >/dev/null 2>&1; then
    kill "${API_PID}" >/dev/null 2>&1 || true
    wait "${API_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

cd "${REPO_ROOT}"
python3 -m src.main api-server \
  --root "${AUTORESEARCH_ROOT}" \
  --port "${API_PORT}" \
  --listen "${API_HOST}" \
  --backend ollama \
  --host http://localhost:11434 \
  --model qwen2.5-coder:7b &
API_PID=$!

for _ in 1 2 3 4 5 6 7 8 9 10; do
  if curl -fsS "${API_BASE}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

curl -fsS "${API_BASE}/health" >/dev/null

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
python3 "${REPO_ROOT}/scripts/run_demo_manager.py" \
  --api-base "${API_BASE}" \
  --packet-path "${DEMO_PACKET}"
