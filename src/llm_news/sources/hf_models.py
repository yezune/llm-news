"""Hugging Face Hub 트렌딩 모델 중 새로 공개된 원본 LLM."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from .. import http
from ..models import Item

API = "https://huggingface.co/api/models"


def fetch(cfg: dict[str, Any], run_date: date) -> list[Item]:
    models = http.get_json(f"{API}?sort=trendingScore&limit={cfg.get('max_candidates', 80)}&full=true")
    tags_ok = set(cfg.get("pipeline_tags", []))
    excluded = [p.lower() for p in cfg.get("exclude_name_patterns", [])]
    cutoff = datetime.combine(run_date, datetime.min.time(), UTC) - timedelta(
        days=cfg.get("max_age_days", 21)
    )
    items = []
    for m in models:
        repo = m["id"]
        created = datetime.fromisoformat(m["createdAt"].replace("Z", "+00:00"))
        tags = m.get("tags", [])
        if tags_ok and m.get("pipeline_tag") not in tags_ok:
            continue
        if created < cutoff or any(p in repo.lower() for p in excluded):
            continue
        if any(t.startswith("base_model:quantized:") for t in tags):
            continue
        arxiv_ids = [t.split(":", 1)[1] for t in tags if t.startswith("arxiv:")]
        items.append(
            Item(
                source="hf_models",
                section="model",
                id=f"hf-model:{repo}",
                title=repo,
                url=f"https://huggingface.co/{repo}",
                published=m["createdAt"],
                summary=f"pipeline: {m.get('pipeline_tag')}; library: {m.get('library_name')}",
                authors=[m.get("author", repo.split("/")[0])],
                signals={
                    "likes": float(m.get("likes", 0)),
                    "trending": float(m.get("trendingScore", 0)),
                    "downloads": float(m.get("downloads", 0)),
                },
                artifacts={"models": [repo], "papers": arxiv_ids},
                tags=[t for t in tags if t.startswith("license:")][:1],
            )
        )
    return items
