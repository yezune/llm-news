"""Hugging Face Daily Papers. 커뮤니티 업보트가 있는 큐레이션된 arXiv 논문 목록."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from .. import http
from ..models import Item

API = "https://huggingface.co/api/daily_papers"


def fetch(cfg: dict[str, Any], run_date: date) -> list[Item]:
    # 실행일 아침에는 전날(UTC) 목록이 완성되어 있다.
    day = run_date - timedelta(days=1)
    entries = http.get_json(f"{API}?date={day.isoformat()}&limit=100")
    entries.sort(key=lambda e: e["paper"].get("upvotes", 0), reverse=True)
    items = []
    for e in entries[: cfg.get("max_candidates", 60)]:
        p = e["paper"]
        items.append(
            Item(
                source="hf_papers",
                section="paper",
                id=f"arxiv:{p['id']}",
                title=p["title"].strip(),
                url=f"https://huggingface.co/papers/{p['id']}",
                published=p.get("publishedAt", ""),
                summary=" ".join(p.get("summary", "").split()),
                authors=[a["name"] for a in p.get("authors", [])],
                signals={"upvotes": float(p.get("upvotes", 0)), "comments": float(e.get("numComments", 0))},
                tags=list(p.get("ai_keywords") or [])[:8],
            )
        )
    return items


def linked_artifacts(arxiv_id: str) -> dict[str, list[str]]:
    """논문 README에서 이 arXiv id를 인용한 HF 저장소. 인기순 상위만."""
    data = http.try_get_json(f"https://huggingface.co/api/arxiv/{arxiv_id}/repos")
    if not data:
        return {}
    out = {}
    for kind in ("models", "datasets"):
        repos = sorted(data.get(kind, []), key=lambda r: r.get("likes", 0), reverse=True)
        out[kind] = [r["id"] for r in repos[:10]]
    return out
