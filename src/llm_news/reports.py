"""주간·월간·연간 리포트: data/의 일일 기록을 기간별로 다시 모아 집계한다.

새로 수집하거나 재검증하지 않는다. 판정과 증거(커밋 sha)는 그날 일일 실행이 남긴 그대로이고,
같은 data/에서는 언제 만들어도 같은 리포트가 나온다(생성 시각을 넣지 않는 이유).
기간이 끝난 뒤 첫 일일 실행이 그 기간 리포트를 확정한다. 진행 중인 기간은 명령줄로 미리 볼 수 있고,
그때 만든 리포트에는 PARTIAL_MARK가 붙어 기간이 끝나면 다시 만들어진다.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .models import Tier, Verdict
from .render import ICON, LABEL, SECTIONS, SOURCE_NAME, _md, stats_line

KINDS = ("week", "month", "year")
KIND_NAME = {"week": "주간", "month": "월간", "year": "연간"}
KIND_DIR = {"week": "weekly", "month": "monthly", "year": "yearly"}
# 섹션별로 싣는 상위 소식 수
TOP_PER_SECTION = {"week": 5, "month": 10, "year": 20}
TIER_NAME = {Tier.INTERNAL: "A 내부 재현", Tier.SOURCE: "B 원본 대조", Tier.REPORTED: "C 자기보고"}
PARTIAL_MARK = "<!-- report:partial -->"


@dataclass(frozen=True)
class Period:
    kind: str
    start: date
    # 마지막 날 포함
    end: date

    @property
    def label(self) -> str:
        if self.kind == "week":
            y, w, _ = self.start.isocalendar()
            return f"{y}-W{w:02d}"
        if self.kind == "month":
            return f"{self.start:%Y-%m}"
        return f"{self.start:%Y}"

    @property
    def relpath(self) -> str:
        if self.kind == "year":
            return f"reports/yearly/{self.label}.md"
        return f"reports/{KIND_DIR[self.kind]}/{self.label[:4]}/{self.label}.md"


def period_of(kind: str, d: date) -> Period:
    if kind == "week":
        start = d - timedelta(days=d.weekday())
        return Period(kind, start, start + timedelta(days=6))
    if kind == "month":
        start = d.replace(day=1)
        return Period(kind, start, (start + timedelta(days=32)).replace(day=1) - timedelta(days=1))
    if kind == "year":
        return Period(kind, date(d.year, 1, 1), date(d.year, 12, 31))
    raise ValueError(f"알 수 없는 기간 종류: {kind}")


def parse_period(kind: str, label: str) -> Period:
    """'2026-W43', '2026-10', '2026' 형식의 기간 이름을 Period로 바꾼다."""
    patterns = {"week": r"(\d{4})-W(\d{2})", "month": r"(\d{4})-(\d{2})", "year": r"(\d{4})"}
    m = re.fullmatch(patterns[kind], label)
    if not m:
        raise ValueError(f"{KIND_NAME[kind]} 기간 형식이 아님: {label}")
    nums = [int(g) for g in m.groups()]
    if kind == "week":
        return period_of(kind, date.fromisocalendar(nums[0], nums[1], 1))
    return period_of(kind, date(nums[0], nums[1] if kind == "month" else 1, 1))


def _segments(p: Period) -> list[tuple[str, date, date]]:
    """흐름 표의 행: 주간은 일별, 월간은 (월 안으로 자른) ISO 주별, 연간은 월별."""
    out, d = [], p.start
    while d <= p.end:
        if p.kind == "week":
            out.append((d.isoformat(), d, d))
            d += timedelta(days=1)
            continue
        sub = period_of("week" if p.kind == "month" else "month", d)
        end = min(sub.end, p.end)
        out.append((sub.label, d, end))
        d = end + timedelta(days=1)
    return out


def data_dates(root: Path) -> list[date]:
    out = []
    for path in (root / "data").glob("*/*.json"):
        try:
            out.append(date.fromisoformat(path.stem))
        except ValueError:
            continue
    return sorted(out)


def load_days(root: Path, start: date, end: date) -> list[dict[str, Any]]:
    days = [d for d in data_dates(root) if start <= d <= end]
    return [json.loads((root / "data" / f"{d:%Y}" / f"{d}.json").read_text(encoding="utf-8")) for d in days]


def _items(days: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """기간 안의 소식. seen.json으로 날짜 간 중복은 이미 없지만, --force 재생성에 대비해 id로 다시 거른다."""
    seen: set[str] = set()
    out = []
    for day in days:
        for it in day["items"]:
            if it["id"] not in seen:
                seen.add(it["id"])
                out.append({**it, "date": day["date"]})
    return out


def _counts(verifications) -> Counter:
    return Counter(Verdict(v["verdict"]) for v in verifications)


def _compact(counts: Counter) -> str:
    return " ".join(f"{ICON[v]}{counts[v]}" for v in Verdict if counts.get(v))


def _digest_link(up: str, d: str, text: str | None = None) -> str:
    return f"[{text or d}]({up}digests/{d[:4]}/{d}.md)"


def render_report(
    period: Period,
    days: list[dict[str, Any]],
    keywords: dict[str, float] | None = None,
    today: date | None = None,
) -> str:
    up = "../" * period.relpath.count("/")
    items = _items(days)
    vs = [(it, v) for it in items for v in it["verifications"]]
    counts = _counts(v for _, v in vs)
    partial = today is not None and period.end >= today

    lines = [PARTIAL_MARK] if partial else []
    lines += [
        f"# LLM {KIND_NAME[period.kind]} 리포트 — {period.label}",
        "",
        f"기간 {period.start} ~ {period.end} · 일일 다이제스트 {len(days)}일 · "
        f"소식 {len(items)}건 · 자동 추출 주장 {len(vs)}건",
        "",
    ]
    if partial:
        last = days[-1]["date"] if days else "기록 없음"
        lines += [
            f"> ⏳ 진행 중인 기간입니다. {last}까지의 기록만 집계했고, "
            "기간이 끝난 뒤 첫 일일 실행에서 확정됩니다.",
            "",
        ]
    if not days:
        return "\n".join(lines + ["이 기간에는 일일 기록이 없습니다."]) + "\n"
    lines += [
        f"**기간 검증:** {stats_line(counts)}",
        "",
        "> 일일 다이제스트의 판정과 증거를 그대로 모은 것입니다. 판정 기준은 "
        f"[METHODOLOGY.md]({up}METHODOLOGY.md) 참고.",
        "",
    ]

    by_date = {d["date"]: d for d in days}
    lines += ["## 📈 흐름", "", "| 구간 | 소식 | 주장 | " + " | ".join(ICON[v] for v in Verdict) + " |"]
    lines.append("|---" * (3 + len(Verdict)) + "|")
    for name, s, e in _segments(period):
        seg = [it for it in items if s.isoformat() <= it["date"] <= e.isoformat()]
        seg_v = [v for it in seg for v in it["verifications"]]
        if not any(s.isoformat() <= d <= e.isoformat() for d in by_date):
            # 기록이 없는 구간(미래 날짜, 실행을 놓친 날)은 0이 아니라 빈칸으로 둔다
            lines.append(f"| {name} |" + " — |" * (2 + len(Verdict)))
            continue
        if period.kind == "week":
            name = _digest_link(up, name)
        c = _counts(seg_v)
        cells = " | ".join(str(c[v]) for v in Verdict)
        lines.append(f"| {name} | {len(seg)} | {len(seg_v)} | {cells} |")
    lines.append("")

    top = TOP_PER_SECTION[period.kind]
    lines += ["## 🏆 주요 소식", ""]
    for section, title in SECTIONS.items():
        sec = sorted(
            (it for it in items if it["section"] == section),
            key=lambda it: (-it["score"], it["date"], it["id"]),
        )[:top]
        if not sec:
            continue
        lines += [f"### {title}", ""]
        for n, it in enumerate(sec, 1):
            src = SOURCE_NAME.get(it["source"], it["source"].removeprefix("rss:"))
            meta = [src, f"점수 {it['score']:g}", _digest_link(up, it["date"], it["date"][5:])]
            if verdicts := _compact(_counts(it["verifications"])):
                meta.append(verdicts)
            lines.append(f"{n}. [{_md(it['title'])}]({it['url']}) — " + " · ".join(meta))
        lines.append("")

    refuted = [(it, v) for it, v in vs if v["verdict"] == Verdict.REFUTED]
    lines += ["## ❌ 반박된 주장", ""]
    if refuted:
        lines += ["| 날짜 | 소식 | 주장 | 근거 |", "|---|---|---|---|"]
        for it, v in refuted:
            ev = next((e for e in v["evidence"] if e.get("url")), None)
            ref = f" [원본]({ev['url']})" if ev else ""
            if ev and ev.get("revision"):
                ref += f" `@{ev['revision'][:7]}`"
            lines.append(
                f"| {_digest_link(up, it['date'], it['date'][5:])} | [{_md(it['title'])}]({it['url']}) | "
                f"{_md(v['claim']['statement'])} | {_md(v['summary'])}{ref} |"
            )
    else:
        lines.append("이 기간 자동 추출 주장 중 반박된 것은 없습니다.")
    lines.append("")

    lines += [
        "## 🔬 검증 계층별 판정",
        "",
        "| 계층 | 주장 | " + " | ".join(f"{ICON[v]} {LABEL[v]}" for v in Verdict) + " |",
        "|---" * (2 + len(Verdict)) + "|",
    ]
    for tier in Tier:
        c = _counts(v for _, v in vs if v["tier"] == tier)
        if c:
            lines.append(
                f"| {TIER_NAME[tier]} | {c.total()} | " + " | ".join(str(c[v]) for v in Verdict) + " |"
            )
    lines.append("")

    if keywords:
        topics = Counter()
        for it in items:
            text = f"{it['title']} {it['summary']} {' '.join(it['tags'])}".lower()
            topics.update(kw for kw in keywords if kw in text)
        if topics:
            lines += ["## 🧩 많이 다룬 주제", "", "| 키워드 | 소식 수 |", "|---|---|"]
            lines += [f"| {kw} | {n} |" for kw, n in topics.most_common(10)]
            lines.append("")

    repos = Counter()
    for it in items:
        for kind in ("models", "datasets"):
            repos.update((kind, r) for r in it["artifacts"].get(kind, []))
    shared = [(k, n) for k, n in repos.most_common(10) if n >= 2]
    if shared:
        lines += ["## 🔗 여러 소식에 연결된 HF 저장소", "", "| 저장소 | 종류 | 소식 수 |", "|---|---|---|"]
        for (kind, repo), n in shared:
            prefix = "datasets/" if kind == "datasets" else ""
            lines.append(
                f"| [{repo}](https://huggingface.co/{prefix}{repo}) | "
                f"{'모델' if kind == 'models' else '데이터셋'} | {n} |"
            )
        lines.append("")

    ledger = days[-1].get("ledger", [])
    drift: dict[tuple[str, str], list[str]] = {}
    for day in days:
        for r in day.get("ledger", []):
            if not r["as_expected"]:
                key = (r["file"], r["verification"]["claim"]["statement"])
                drift.setdefault(key, []).append(day["date"])
    lines += ["## 🧮 검증 대장 (claims/)", ""]
    if ledger:
        ledger_stats = stats_line(_counts(r["verification"] for r in ledger))
        lines += [f"마지막 기록({days[-1]['date']}) 기준 명세 {len(ledger)}건 · {ledger_stats}", ""]
    if drift:
        lines += ["기대 판정에서 벗어난 명세 (원본 아티팩트가 바뀌었는지 확인 필요):", ""]
        lines += [
            f"- `{file}` {_md(stmt)} — {len(ds)}일 ({ds[0]} ~ {ds[-1]})" for (file, stmt), ds in drift.items()
        ]
    else:
        lines.append("기간 내내 모든 명세가 기대 판정과 같았습니다.")
    lines.append("")

    errors = Counter(name for day in days for name in day.get("errors", {}))
    if errors:
        lines += ["## ⚠️ 수집 오류", ""]
        lines += [f"- `{name}`: {n}일" for name, n in sorted(errors.items())]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def is_final(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as f:
        return f.readline().strip() != PARTIAL_MARK


def due_periods(root: Path, run_date: date) -> list[Period]:
    """기록이 있고 run_date 전에 끝났는데 확정 리포트가 없는 기간. 놓친 날이 있어도 따라잡는다."""
    dates = data_dates(root)
    due = []
    for kind in KINDS:
        for p in sorted({period_of(kind, d) for d in dates}, key=lambda p: p.start):
            if p.end < run_date and not is_final(root / p.relpath):
                due.append(p)
    return due


def write_report(root: Path, period: Period, config: dict[str, Any], today: date) -> Path:
    path = root / period.relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    days = load_days(root, period.start, period.end)
    path.write_text(render_report(period, days, config.get("keywords"), today), encoding="utf-8")
    return path


def latest_reports(root: Path) -> dict[str, Path]:
    out = {}
    for kind in KINDS:
        files = sorted((root / "reports" / KIND_DIR[kind]).glob("**/*.md"), key=lambda p: p.stem)
        if files:
            out[kind] = files[-1]
    return out
