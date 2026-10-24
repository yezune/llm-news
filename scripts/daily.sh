#!/usr/bin/env bash
# GitHub Actions 대신 로컬(cron/launchd)에서 하루 1회 돌릴 때 쓰는 진입점.
#   crontab 예: 17 7 * * *  /path/to/llm-news/scripts/daily.sh >> /tmp/llm-news.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --frozen --extra dev --quiet
uv run pytest -q
uv run llm-news -v daily "$@"
uv run llm-news ledger --check > /dev/null || echo "경고: claims/ 판정이 기대값과 다릅니다" >&2
