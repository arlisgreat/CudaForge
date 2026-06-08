#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/streamma_checkpoint.sh <stage-name>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="$1"
SAFE_LABEL="$(printf "%s" "$LABEL" | tr -cs 'A-Za-z0-9_.-' '_' | sed 's/^_//; s/_$//')"
STAMP="$(date +%Y%m%d_%H%M%S)"
TAG="streamma/${SAFE_LABEL}-${STAMP}"

cd "$ROOT"

if ! git diff --quiet; then
  echo "working tree has unstaged changes; commit or discard them before tagging" >&2
  exit 1
fi

if ! git diff --cached --quiet; then
  echo "index has staged changes; commit or unstage them before tagging" >&2
  exit 1
fi

git tag -a "$TAG" -m "StreamMA checkpoint: $LABEL"
echo "$TAG"
