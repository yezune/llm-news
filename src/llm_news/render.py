"""다이제스트(Markdown) 렌더링."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .models import Item, Verdict, Verification
from .specs import SpecResult

ICON = {
    Verdict.VERIFIED: "✅",
    Verdict.REFUTED: "❌",
    Verdict.ATTESTED: "📄",
    Verdict.INCONCLUSIVE: "❔",
    Verdict.UNVERIFIABLE: "⚪",
}
LABEL = {
    Verdict.VERIFIED: "검증됨",
    Verdict.REFUTED: "반박됨",
    Verdict.ATTESTED: "원본 선언 일치",
    Verdict.INCONCLUSIVE: "판단 보류",
    Verdict.UNVERIFIABLE: "검증 불가(자기보고)",
}
SECTIONS = {"paper": "📄 논문", "model": "🤖 새 모델", "blog": "📰 연구소 소식"}
SOURCE_NAME = {"hf_papers": "HF Daily Papers", "arxiv": "arXiv", "hf_models": "HF Hub 트렌딩"}


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def verdict_counts(verifications: Iterable[Verification]) -> Counter:
    return Counter(v.verdict for v in verifications)


def stats_line(counts: Counter) -> str:
    parts = [f"{ICON[v]} {LABEL[v]} {counts[v]}" for v in Verdict if counts.get(v)]
    return " · ".join(parts) if parts else "검증 대상 주장 없음"


def _verification_table(vs: list[Verification]) -> list[str]:
    lines = ["", "| 판정 | 주장 | 방법 | 근거 |", "|---|---|---|---|"]
    for v in vs:
        links = [e.url for e in v.evidence if e.url]
        ref = f"[원본]({links[0]})" if links else ""
        if v.evidence and v.evidence[0].revision:
            ref += f" `@{v.evidence[0].revision[:7]}`"
        lines.append(
            f"| {ICON[v.verdict]} {LABEL[v.verdict]} | {_md(v.claim.statement)} | "
            f"`{v.method}` ({v.tier.value}) | {_md(v.summary)} {ref} |"
        )
    return lines


def _signals(item: Item) -> str:
    s = item.signals
    bits = []
    if "upvotes" in s:
        bits.append(f"▲{int(s['upvotes'])}")
    if "likes" in s:
        bits.append(f"♥{int(s['likes'])}")
    if "downloads" in s:
        bits.append(f"⬇{int(s['downloads'])}")
    return " ".join(bits)


def render_item(n: int, item: Item) -> list[str]:
    src = SOURCE_NAME.get(item.source, item.source.removeprefix("rss:"))
    head = f"### {n}. [{_md(item.title)}]({item.url})"
    meta = [f"출처: {src}", f"점수 {item.score:g}"]
    if sig := _signals(item):
        meta.append(sig)
    if item.published:
        meta.append(item.published[:10])
    lines = [head, "", "_" + " · ".join(meta) + "_", ""]
    if item.authors and item.section == "paper":
        authors = ", ".join(item.authors[:4]) + (" 외" if len(item.authors) > 4 else "")
        lines += [f"저자: {authors}", ""]
    if item.summary and item.section != "model":
        summary = item.summary if len(item.summary) <= 420 else item.summary[:420].rsplit(" ", 1)[0] + " …"
        lines += [f"> {summary}", ""]
    arts = [(k, v) for k, v in item.artifacts.items() if v and k in ("models", "datasets")]
    if arts and item.section != "model":
        for kind, repos in arts:
            prefix = "datasets/" if kind == "datasets" else ""
            links = ", ".join(f"[{r}](https://huggingface.co/{prefix}{r})" for r in repos[:5])
            lines.append(f"- 연결된 {'모델' if kind == 'models' else '데이터셋'}: {links}")
        lines.append("")
    if "math-candidate" in item.tags:
        lines += ["- 🧮 수학적 주장 포함 — `claims/`에 명세를 추가하면 내부 검증됩니다.", ""]
    if item.verifications:
        lines += _verification_table(item.verifications)
        lines.append("")
    return lines


def render_digest(
    run_date: str,
    generated_at: str,
    items: list[Item],
    ledger: list[SpecResult],
    errors: dict[str, str],
) -> str:
    all_v = [v for it in items for v in it.verifications]
    lines = [
        f"# LLM 데일리 — {run_date}",
        "",
        f"생성 {generated_at} · 소식 {len(items)}건 · 자동 추출 주장 {len(all_v)}건",
        "",
        f"**오늘의 검증:** {stats_line(verdict_counts(all_v))}",
        "",
        "> 판정 기준은 [METHODOLOGY.md](../../METHODOLOGY.md) 참고. "
        "✅는 내부 재현 또는 원본 아티팩트 대조로 확인된 것, ⚪는 근거가 없어 자기보고로 남긴 것입니다.",
        "",
    ]
    for section, title in SECTIONS.items():
        sec = [it for it in items if it.section == section]
        if not sec:
            continue
        lines += [f"## {title}", ""]
        for i, it in enumerate(sec, 1):
            lines += render_item(i, it)
    if ledger:
        ledger_stats = stats_line(verdict_counts(r.verification for r in ledger))
        lines += [
            "## 🧮 검증 대장 (claims/)",
            "",
            f"사람이 작성한 명세 {len(ledger)}건 · {ledger_stats}",
            "",
            "| 판정 | 대상 | 주장 | 방법 | 결과 |",
            "|---|---|---|---|---|",
        ]
        for r in ledger:
            v = r.verification
            drift = "" if r.as_expected else f" ⚠️ 기대 판정 `{r.expected}`과 다름"
            title = f"[{_md(r.title)}]({r.source})" if r.source else _md(r.title)
            lines.append(
                f"| {ICON[v.verdict]} | {title} | {_md(v.claim.statement)} | `{v.method}` | "
                f"{_md(v.summary)}{drift} |"
            )
        lines.append("")
    if errors:
        lines += ["## ⚠️ 수집 오류", ""]
        lines += [f"- `{name}`: {_md(msg)}" for name, msg in errors.items()]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
