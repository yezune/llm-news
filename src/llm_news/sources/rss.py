"""연구소 블로그 RSS/Atom 피드."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import UTC, date, datetime, timedelta
from typing import Any

import feedparser

from .. import http
from ..models import Item

log = logging.getLogger(__name__)


def _strip_html(text: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", text or "").split())


def fetch(cfg: dict[str, Any], run_date: date) -> list[Item]:
    since = datetime.combine(run_date - timedelta(days=cfg.get("window_days", 3)), datetime.min.time(), UTC)
    items = []
    for feed_cfg in cfg.get("feeds", []):
        try:
            feed = feedparser.parse(http.get_text(feed_cfg["url"]))
        except Exception as exc:  # 피드 하나가 죽어도 나머지는 수집
            log.warning("feed %s failed: %s", feed_cfg["name"], exc)
            continue
        for e in feed.entries:
            stamp = e.get("published_parsed") or e.get("updated_parsed")
            if not stamp:
                continue
            published = datetime(*stamp[:6], tzinfo=UTC)
            if published < since:
                continue
            link = e.get("link", "")
            items.append(
                Item(
                    source=f"rss:{feed_cfg['name']}",
                    section="blog",
                    id=f"url:{hashlib.sha1(link.encode()).hexdigest()[:16]}",
                    title=" ".join(e.get("title", "").split()),
                    url=link,
                    published=published.isoformat(),
                    summary=_strip_html(e.get("summary", ""))[:1200],
                    authors=[feed_cfg["name"]],
                )
            )
    return items
