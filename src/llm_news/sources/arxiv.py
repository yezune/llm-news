"""arXiv API (Atom). HF Daily Papers가 놓친 논문을 보완한다."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import feedparser

from .. import http
from ..models import Item

API = "https://export.arxiv.org/api/query"


def fetch(cfg: dict[str, Any], run_date: date) -> list[Item]:
    query = urlencode(
        {
            "search_query": cfg["query"],
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": cfg.get("max_results", 100),
        }
    )
    feed = feedparser.parse(http.get_text(f"{API}?{query}"))
    since = datetime.combine(run_date - timedelta(days=cfg.get("window_days", 2)), datetime.min.time(), UTC)
    items = []
    for e in feed.entries:
        published = datetime(*e.published_parsed[:6], tzinfo=UTC)
        if published < since:
            continue
        # http://arxiv.org/abs/2610.12345v1 -> 2610.12345
        arxiv_id = re.sub(r"v\d+$", "", e.id.rsplit("/abs/", 1)[-1])
        items.append(
            Item(
                source="arxiv",
                section="paper",
                id=f"arxiv:{arxiv_id}",
                title=" ".join(e.title.split()),
                url=f"https://arxiv.org/abs/{arxiv_id}",
                published=published.isoformat(),
                summary=" ".join(e.summary.split()),
                authors=[a.name for a in e.get("authors", [])],
                tags=[t["term"] for t in e.get("tags", [])][:5],
            )
        )
    return items
