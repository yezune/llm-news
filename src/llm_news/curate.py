"""큐레이션: 관련도 필터 -> 중복 병합 -> 이전 날짜에 실린 항목 제외 -> 점수순 섹션별 선택.

점수 = 출처 가중치 × (1 + min(relevance, 상한)/2) + Σ 신호 가중치 × log1p(인기 신호).
relevance는 'LLM 소식인가'를 거르는 용도라 상한을 둔다 — 키워드를 많이 쓴 글이 아니라
사람들이 실제로 주목한 글(업보트·좋아요)이 위로 오게 한다. 결정론적이며 설정 파일로만 조정한다.
"""

from __future__ import annotations

import math
from typing import Any

from .models import Item

# 인기 신호별 가중치. 업보트는 사람이 고른 신호라 가장 크게 본다.
SIGNAL_WEIGHTS = {"upvotes": 2.0, "likes": 0.6, "trending": 0.4, "comments": 0.3}
RELEVANCE_CAP = 6.0


def relevance(item: Item, keywords: dict[str, float]) -> float:
    text = f"{item.title} {item.summary} {' '.join(item.tags)}".lower()
    return sum(w for kw, w in keywords.items() if kw in text)


def requires_keywords(item: Item, config: dict[str, Any]) -> bool:
    sources = config.get("sources", {})
    if item.source.startswith("rss:"):
        name = item.source.split(":", 1)[1]
        feed = next((f for f in sources.get("rss", {}).get("feeds", []) if f["name"] == name), {})
        return bool(feed.get("require_keywords", False))
    return bool(sources.get(item.source, {}).get("require_keywords", False))


def source_weight(item: Item, config: dict[str, Any]) -> float:
    key = "rss" if item.source.startswith("rss:") else item.source
    return float(config.get("sources", {}).get(key, {}).get("weight", 1.0))


def merge_duplicates(items: list[Item]) -> list[Item]:
    """같은 정규 id(예: 같은 arXiv 논문)를 하나로 합친다. 인기 신호가 있는 쪽을 대표로 쓴다."""
    by_id: dict[str, Item] = {}
    for it in items:
        cur = by_id.get(it.id)
        if cur is None:
            by_id[it.id] = it
            continue
        primary, other = (cur, it) if sum(cur.signals.values()) >= sum(it.signals.values()) else (it, cur)
        primary.signals = {**other.signals, **primary.signals}
        primary.summary = primary.summary or other.summary
        primary.tags = list(dict.fromkeys(primary.tags + other.tags))
        for kind, repos in other.artifacts.items():
            primary.artifacts[kind] = list(dict.fromkeys(primary.artifacts.get(kind, []) + repos))
        by_id[it.id] = primary
    return list(by_id.values())


def score(item: Item, config: dict[str, Any]) -> float:
    rel = relevance(item, config.get("keywords", {}))
    pop = sum(SIGNAL_WEIGHTS.get(k, 0) * math.log1p(v) for k, v in item.signals.items())
    return round(source_weight(item, config) * (1 + min(rel, RELEVANCE_CAP) / 2) + pop, 3)


def curate(items: list[Item], config: dict[str, Any], seen: dict[str, str], run_date: str) -> list[Item]:
    """seen: {item_id: 처음 실린 날짜}. 오늘 이미 실린 항목(재실행)은 유지한다."""
    keywords = config.get("keywords", {})
    min_rel = float(config.get("min_relevance", 0))
    pool = []
    for it in merge_duplicates(items):
        first = seen.get(it.id)
        if first is not None and first != run_date:
            continue
        if requires_keywords(it, config) and relevance(it, keywords) < min_rel:
            continue
        it.score = score(it, config)
        pool.append(it)
    limits = config.get("limits", {})
    per_source = limits.get("max_per_source", {})
    selected = []
    for section in ("paper", "model", "blog"):
        ranked = sorted((it for it in pool if it.section == section), key=lambda x: (-x.score, x.id))
        cap = per_source.get(section)
        counts: dict[str, int] = {}
        picked = []
        for it in ranked:
            if len(picked) >= int(limits.get(section, 10)):
                break
            if cap is not None and counts.get(it.source, 0) >= int(cap):
                continue
            counts[it.source] = counts.get(it.source, 0) + 1
            picked.append(it)
        selected.extend(picked)
    return selected
