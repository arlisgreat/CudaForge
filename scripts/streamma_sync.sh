#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${STREAMMA_REMOTE:-origin}"
BRANCH="${STREAMMA_BRANCH:-$(git -C "$ROOT" branch --show-current)}"

if [[ -z "$BRANCH" ]]; then
  echo "cannot determine current branch; set STREAMMA_BRANCH" >&2
  exit 2
fi

cd "$ROOT"

if ! git diff --quiet; then
  echo "working tree has unstaged changes; commit or discard before syncing" >&2
  exit 1
fi

if ! git diff --cached --quiet; then
  echo "index has staged changes; commit or unstage before syncing" >&2
  exit 1
fi

git push -u "$REMOTE" "HEAD:$BRANCH"
git push "$REMOTE" --tags
