"""일일 파이프라인: 수집 -> 큐레이션 -> 아티팩트 연결 -> 주장 추출 -> 검증 -> 기록."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from . import http
from .curate import curate
from .extract import extract
from .models import Item
from .render import render_digest, stats_line, verdict_counts
from .reports import KIND_NAME, due_periods, is_final, latest_reports, write_report
from .sources import fetch_all
from .sources.hf_papers import linked_artifacts
from .specs import SpecResult, run_specs
from .verify import hf, verify
from .verify.algorithms import load_reproductions

log = logging.getLogger(__name__)

README_START = "<!-- LATEST:START -->"
README_END = "<!-- LATEST:END -->"
REPORTS_START = "<!-- REPORTS:START -->"
REPORTS_END = "<!-- REPORTS:END -->"


class Paths:
    def __init__(self, root: Path):
        self.root = root
        self.config = root / "config" / "sources.yaml"
        self.claims = root / "claims"
        self.reproductions = root / "reproductions"
        self.state = root / "state" / "seen.json"
        self.readme = root / "README.md"

    def digest(self, d: str) -> Path:
        return self.root / "digests" / d[:4] / f"{d}.md"

    def data(self, d: str) -> Path:
        return self.root / "data" / d[:4] / f"{d}.json"


def load_config(paths: Paths) -> dict[str, Any]:
    return yaml.safe_load(paths.config.read_text(encoding="utf-8"))


def today(config: dict[str, Any]) -> date:
    return datetime.now(ZoneInfo(config.get("timezone", "UTC"))).date()


def enrich(items: list[Item]) -> dict[str, str]:
    """원본 대조 검증의 연결고리를 붙인다.

    - 논문: 그 arXiv id를 인용한 HF 모델/데이터셋
    - 모델: 저자가 선언한 수치를 담은 모델 카드(README) 본문 → {item.id: card}
    """
    cards: dict[str, str] = {}
    for it in items:
        if it.section == "paper" and not (it.artifacts.get("models") or it.artifacts.get("datasets")):
            it.artifacts.update(linked_artifacts(it.id.removeprefix("arxiv:")))
        elif it.section == "model" and it.artifacts.get("models"):
            repo = it.artifacts["models"][0]
            info = hf.model_info(repo)
            card = hf.model_card(repo, info["sha"]) if info and info.get("sha") else None
            if card:
                cards[it.id] = card
    return cards


def verify_items(items: list[Item], cards: dict[str, str] | None = None) -> None:
    for it in items:
        it.claims = extract(it, (cards or {}).get(it.id))
        it.verifications = [verify(c) for c in it.claims]


def run_ledger(paths: Paths) -> list[SpecResult]:
    if paths.reproductions.exists():
        load_reproductions(paths.reproductions)
    return run_specs(paths.claims) if paths.claims.exists() else []


def _ledger_json(results: list[SpecResult]) -> list[dict[str, Any]]:
    return [
        {
            "file": r.file,
            "title": r.title,
            "source": r.source,
            "item": r.item,
            "expected": r.expected,
            "as_expected": r.as_expected,
            "verification": asdict(r.verification),
        }
        for r in results
    ]


def _replace_block(paths: Paths, start: str, end: str, body: str) -> None:
    if not paths.readme.exists():
        return
    text = paths.readme.read_text(encoding="utf-8")
    if start not in text:
        return
    block = f"{start}\n{body}\n{end}"
    text = re.sub(re.escape(start) + ".*?" + re.escape(end), lambda _: block, text, flags=re.S)
    paths.readme.write_text(text, encoding="utf-8")


def update_readme(paths: Paths, run_date: str, summary: str) -> None:
    digests = sorted(paths.root.glob("digests/*/*.md"), reverse=True)[:14]
    recent = "\n".join(f"- [{p.stem}](digests/{p.parent.name}/{p.name})" for p in digests)
    body = (
        f"**최신 다이제스트: [{run_date}](digests/{run_date[:4]}/{run_date}.md)** — {summary}\n\n"
        f"최근 14일:\n\n{recent}"
    )
    _replace_block(paths, README_START, README_END, body)


def update_readme_reports(paths: Paths) -> None:
    latest = latest_reports(paths.root)
    links = [
        f"{KIND_NAME[kind]} [{p.stem}]({p.relative_to(paths.root).as_posix()})"
        + ("" if is_final(p) else " ⏳")
        for kind, p in latest.items()
    ]
    body = "**기간 리포트:** " + (" · ".join(links) if links else "아직 없음 (기간이 끝나면 자동 생성)")
    _replace_block(paths, REPORTS_START, REPORTS_END, body)


def run_reports(root: Path, run_date: date | None = None) -> list[Path]:
    """끝난 기간(주·월·년)의 리포트를 확정한다. 이미 확정된 것은 건드리지 않으므로 매일 불러도 된다."""
    paths = Paths(root)
    config = load_config(paths)
    run_date = run_date or today(config)
    out = [write_report(root, p, config, run_date) for p in due_periods(root, run_date)]
    update_readme_reports(paths)
    return out


def run_daily(root: Path, run_date: date | None = None, force: bool = False) -> Path | None:
    """하루 1회 실행. 같은 날짜의 다이제스트가 있으면 force 없이는 건너뛴다(None 반환)."""
    paths = Paths(root)
    config = load_config(paths)
    run_date = run_date or today(config)
    d = run_date.isoformat()
    if paths.digest(d).exists() and not force:
        log.info("digest for %s already exists; skipping (use --force)", d)
        return None

    seen: dict[str, str] = json.loads(paths.state.read_text()) if paths.state.exists() else {}
    raw, errors = fetch_all(config, run_date)
    items = curate(raw, config, seen, d)
    cards = enrich(items)
    verify_items(items, cards)
    ledger = run_ledger(paths)
    generated_at = http.now_iso()

    paths.digest(d).parent.mkdir(parents=True, exist_ok=True)
    paths.digest(d).write_text(render_digest(d, generated_at, items, ledger, errors), encoding="utf-8")
    paths.data(d).parent.mkdir(parents=True, exist_ok=True)
    paths.data(d).write_text(
        json.dumps(
            {
                "date": d,
                "generated_at": generated_at,
                "fetched": len(raw),
                "errors": errors,
                "items": [asdict(it) for it in items],
                "ledger": _ledger_json(ledger),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    for it in items:
        seen.setdefault(it.id, d)
    paths.state.parent.mkdir(parents=True, exist_ok=True)
    paths.state.write_text(json.dumps(dict(sorted(seen.items())), indent=0, ensure_ascii=False))

    summary = stats_line(verdict_counts(v for it in items for v in it.verifications))
    update_readme(paths, d, f"소식 {len(items)}건 · {summary}")
    return paths.digest(d)
