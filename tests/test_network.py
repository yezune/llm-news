"""실제 Hugging Face에 접속해 검증 대장 전체를 기대 판정과 대조한다.

  pytest -m network
원본 저장소가 바뀌어 판정이 달라지면 실패한다 — 그것이 이 테스트의 목적이다(드리프트 감지).
"""

import pytest
from conftest import ROOT

from llm_news.pipeline import Paths, run_ledger

pytestmark = pytest.mark.network


def test_full_ledger_matches_expected_verdicts_against_live_sources():
    results = run_ledger(Paths(ROOT))
    drifted = [
        (r.file, r.verification.claim.statement, r.verification.verdict, r.expected, r.verification.summary)
        for r in results
        if not r.as_expected
    ]
    assert not drifted
