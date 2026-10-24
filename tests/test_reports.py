import json
import shutil
from dataclasses import asdict
from datetime import date

import pytest
from conftest import ROOT

from llm_news.models import Claim, Evidence, Item, Tier, Verdict, Verification
from llm_news.pipeline import REPORTS_END, REPORTS_START, run_reports
from llm_news.reports import PARTIAL_MARK, due_periods, is_final, parse_period, period_of, write_report


@pytest.mark.parametrize(
    "kind,d,label,start,end",
    [
        ("week", date(2026, 10, 23), "2026-W43", date(2026, 10, 19), date(2026, 10, 25)),
        # ISO 주는 연도를 넘는다: 2027-01-01(금)은 2026년의 53번째 주
        ("week", date(2027, 1, 1), "2026-W53", date(2026, 12, 28), date(2027, 1, 3)),
        ("month", date(2026, 2, 14), "2026-02", date(2026, 2, 1), date(2026, 2, 28)),
        ("month", date(2026, 12, 31), "2026-12", date(2026, 12, 1), date(2026, 12, 31)),
        ("year", date(2026, 10, 23), "2026", date(2026, 1, 1), date(2026, 12, 31)),
    ],
)
def test_period_boundaries_and_labels(kind, d, label, start, end):
    p = period_of(kind, d)
    assert (p.label, p.start, p.end) == (label, start, end)
    assert parse_period(kind, label) == p


def test_parse_period_rejects_wrong_format():
    with pytest.raises(ValueError):
        parse_period("week", "2026-10")


def item(id, title, section="paper", score=1.0, verdicts=(), repos=()):
    vs = [
        Verification(
            Claim("hf_model_params", f"{title} 주장 {i}", {"repo": "org/m"}),
            verdict,
            Tier.SOURCE if verdict != Verdict.UNVERIFIABLE else Tier.REPORTED,
            "hf_model_params",
            "실측 8.54B" if verdict == Verdict.REFUTED else "일치",
            [Evidence("safetensors", url="https://huggingface.co/org/m", revision="deadbeef1234")],
        )
        for i, verdict in enumerate(verdicts)
    ]
    return Item(
        "hf_papers",
        section,
        id,
        title,
        f"https://x/{id}",
        "",
        summary="An LLM paper",
        score=score,
        artifacts={"models": list(repos)},
        verifications=vs,
    )


def write_day(root, d, items, errors=None, drift=False):
    path = root / "data" / d[:4] / f"{d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger_v = asdict(
        Verification(Claim("math_value", "e^2", {}), Verdict.VERIFIED, Tier.INTERNAL, "sympy", "ok")
    )
    ledger = [
        {"file": "m.yaml", "expected": "refuted" if drift else "verified", "as_expected": not drift}
        | {"title": "t", "source": None, "item": None, "verification": ledger_v}
    ]
    doc = {"date": d, "items": [asdict(i) for i in items], "errors": errors or {}, "ledger": ledger}
    path.write_text(json.dumps(doc, ensure_ascii=False))


VERIFIED, REFUTED = Verdict.VERIFIED, Verdict.REFUTED


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    shutil.copy(ROOT / "config" / "sources.yaml", root / "config" / "sources.yaml")
    (root / "README.md").write_text(f"# t\n{REPORTS_START}\nold\n{REPORTS_END}\n")
    write_day(
        root,
        "2026-10-23",
        [
            item("arxiv:1", "Gemma claim", score=5, verdicts=[VERIFIED, REFUTED], repos=["o/a"]),
            item("hf-model:o/a", "o/a", section="model", score=3, verdicts=[VERIFIED]),
        ],
        errors={"arxiv": "HTTP 429"},
    )
    write_day(
        root,
        "2026-10-25",
        [item("arxiv:2", "Top paper", score=9, verdicts=[Verdict.UNVERIFIABLE], repos=["o/a"])],
        drift=True,
    )
    return root


def test_weekly_report_aggregates_days(repo):
    path = write_report(repo, parse_period("week", "2026-W43"), {"keywords": {"llm": 1}}, date(2026, 10, 26))
    md = path.read_text()
    assert path == repo / "reports" / "weekly" / "2026" / "2026-W43.md"
    assert PARTIAL_MARK not in md and is_final(path)
    assert "일일 다이제스트 2일 · 소식 3건 · 자동 추출 주장 4건" in md
    # 점수순, 날짜 간 합산
    assert md.index("Top paper") < md.index("Gemma claim")
    # 반박된 주장은 근거와 고정 커밋까지 빠짐없이
    assert "Gemma claim 주장 1" in md and "실측 8.54B" in md and "`@deadbee`" in md
    # 기록 없는 날은 0이 아닌 빈칸, 기록 있는 날은 일일 다이제스트 링크
    assert "| 2026-10-24 | — |" in md
    assert "[2026-10-23](../../../digests/2026/2026-10-23.md) | 2 | 3 |" in md
    assert "| [o/a](https://huggingface.co/o/a) | 모델 | 2 |" in md
    assert "| llm | 3 |" in md
    assert "`m.yaml` e^2 — 1일 (2026-10-25 ~ 2026-10-25)" in md
    assert "`arxiv`: 1일" in md


def test_report_is_deterministic(repo):
    p = parse_period("month", "2026-10")
    first = write_report(repo, p, {}, date(2026, 11, 1)).read_text()
    assert write_report(repo, p, {}, date(2026, 11, 1)).read_text() == first
    assert "| 2026-W43 | 3 | 4 |" in first and "| 2026-W44 | — |" in first


def test_in_progress_report_is_marked_partial(repo):
    path = write_report(repo, period_of("year", date(2026, 10, 26)), {}, date(2026, 10, 26))
    assert path == repo / "reports" / "yearly" / "2026.md"
    assert path.read_text().startswith(PARTIAL_MARK) and not is_final(path)
    assert "2026-10-25까지의 기록만 집계" in path.read_text()


def test_due_periods_finalize_after_period_ends(repo):
    # 일요일(주 마지막 날)에는 아직 확정하지 않는다
    assert due_periods(repo, date(2026, 10, 25)) == []
    # 월요일 실행이 지난주를 확정
    assert [p.label for p in due_periods(repo, date(2026, 10, 26))] == ["2026-W43"]
    assert [p.label for p in due_periods(repo, date(2026, 11, 1))] == ["2026-W43", "2026-10"]
    assert [p.label for p in due_periods(repo, date(2027, 1, 1))] == ["2026-W43", "2026-10", "2026"]


def test_run_reports_is_idempotent_and_replaces_previews(repo):
    preview = write_report(repo, parse_period("week", "2026-W43"), {}, date(2026, 10, 24))
    assert not is_final(preview)
    assert run_reports(repo, date(2026, 10, 26)) == [preview]
    assert is_final(preview)
    assert run_reports(repo, date(2026, 10, 27)) == []
    readme = (repo / "README.md").read_text()
    assert "주간 [2026-W43](reports/weekly/2026/2026-W43.md)" in readme and "old" not in readme
