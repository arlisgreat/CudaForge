#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="/data/workspace/airulan/conda_envs/kernelbench_py310_cu124/bin/python"

cd "$ROOT"

echo "repo=$ROOT"
echo "branch=$(git branch --show-current 2>/dev/null || true)"
echo "head=$(git rev-parse --short HEAD 2>/dev/null || true)"
echo

echo "[git status]"
git status --short
echo

echo "[recent commits]"
git log --oneline --decorate -5
echo

echo "[remotes]"
git remote -v
echo

echo "[python]"
if [[ -x "$PY" ]]; then
  "$PY" --version
else
  python --version
fi
echo

echo "[env files]"
for env_file in /data1/workspace/airulan/env124.sh /data1/workspace/airulan/env130.sh; do
  if [[ -f "$env_file" ]]; then
    echo "present $env_file"
  else
    echo "missing $env_file"
  fi
done
echo

echo "[gpu]"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -8
else
  echo "nvidia-smi not found"
fi
