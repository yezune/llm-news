"""소식 출처 수집기. 각 모듈은 fetch(cfg, run_date) -> list[Item] 을 제공한다."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from ..models import Item
from . import arxiv, hf_models, hf_papers, rss

log = logging.getLogger(__name__)

FETCHERS = {
    "hf_papers": hf_papers.fetch,
    "arxiv": arxiv.fetch,
    "hf_models": hf_models.fetch,
    "rss": rss.fetch,
}


def fetch_all(config: dict[str, Any], run_date: date) -> tuple[list[Item], dict[str, str]]:
    """모든 출처를 수집한다. 한 출처의 실패가 전체 실행을 막지 않도록 오류는 따로 모은다."""
    items: list[Item] = []
    errors: dict[str, str] = {}
    for name, cfg in config.get("sources", {}).items():
        if not cfg.get("enabled", True):
            continue
        try:
            got = FETCHERS[name](cfg, run_date)
        except Exception as exc:
            log.warning("source %s failed: %s", name, exc)
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        log.info("source %s: %d items", name, len(got))
        items.extend(got)
    return items, errors
