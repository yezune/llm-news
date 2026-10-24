import json
import shutil
from datetime import date

import pytest
from conftest import HUB, ROOT, add_model, fixture_json

from llm_news.models import Verdict
from llm_news.pipeline import README_END, README_START, run_daily
from llm_news.specs import load_spec, run_specs
from llm_news.verify import VERIFIERS
from llm_news.verify.algorithms import load_reproductions

RUN = date(2026, 10, 23)

# 네트워크 없이 저장소 안에서 끝나는 검증 종류 (Tier A)
INTERNAL_KINDS = {"math_identity", "math_inequality", "math_value", "algorithm", "kv_cache_bytes"}


def all_specs():
    return sorted((ROOT / "claims").glob("**/*.yaml"))


@pytest.mark.parametrize("path", all_specs(), ids=lambda p: p.name)
def test_every_spec_is_well_formed_and_has_a_verifier(path):
    _, claims = load_spec(path)
    assert claims
    for claim, _expected in claims:
        assert claim.kind in VERIFIERS, f"{path}: 검증기 없는 종류 {claim.kind}"


def test_internal_ledger_claims_match_expected_verdicts(tmp_path, web):
    """수학·알고리즘·산술 주장은 CI에서 네트워크 없이 전부 재검증한다."""
    load_reproductions(ROOT / "reproductions")
    internal = tmp_path / "claims"
    internal.mkdir()
    for path in all_specs():
        doc, claims = load_spec(path)
        keep = [
            c
            for c in doc["claims"]
            if c["kind"] in INTERNAL_KINDS or (c["kind"] == "training_flops" and "params" in c["params"])
        ]
        if keep:
            (internal / path.name).write_text(json.dumps({**doc, "claims": keep}))
    results = run_specs(internal)
    assert len(results) >= 15
    drifted = [
        (r.verification.claim.statement, r.verification.verdict, r.expected)
        for r in results
        if not r.as_expected
    ]
    assert not drifted


def make_repo(tmp_path):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    shutil.copy(ROOT / "config" / "sources.yaml", root / "config" / "sources.yaml")
    (root / "README.md").write_text(f"# t\n{README_START}\nold\n{README_END}\n")
    claims = root / "claims"
    claims.mkdir()
    (claims / "m.yaml").write_text(
        "title: t\nclaims:\n  - kind: math_value\n    statement: e^2\n"
        "    params: {expr: exp(2), value: 7.389056}\n    expected: verified\n"
    )
    return root


def stub_sources(web):
    web.add(
        f"{HUB}/api/daily_papers?date=2026-10-22&limit=100",
        [
            {
                "paper": {
                    "id": "2610.00001",
                    "title": "A 0.5B parameter LLM",
                    "upvotes": 50,
                    "summary": "We release a 0.5B parameter large language model.",
                    "authors": [{"name": "Kim"}],
                    "publishedAt": "2026-10-21T00:00:00Z",
                },
                "numComments": 1,
            },
            {
                "paper": {
                    "id": "2610.00002",
                    "title": "Robot hands",
                    "upvotes": 999,
                    "summary": "Dexterous manipulation.",
                    "authors": [],
                }
            },
        ],
    )
    web.add(
        f"{HUB}/api/arxiv/2610.00001/repos",
        {"models": [{"id": "Qwen/Qwen2.5-0.5B", "likes": 1}], "datasets": []},
    )
    add_model(web, "Qwen/Qwen2.5-0.5B", total=494_032_768, config=fixture_json("qwen2.5-0.5b.config.json"))
    web.add(f"{HUB}/api/models?sort=trendingScore&limit=80&full=true", [])
    # arxiv, rss는 실패 → 수집 오류로 기록되고 파이프라인은 계속


def test_daily_pipeline_end_to_end(tmp_path, web):
    root = make_repo(tmp_path)
    stub_sources(web)
    out = run_daily(root, RUN)
    assert out == root / "digests" / "2026" / "2026-10-23.md"
    md = out.read_text()
    assert "A 0.5B parameter LLM" in md and "Robot hands" not in md
    assert "✅ 검증됨" in md and "Qwen/Qwen2.5-0.5B" in md
    assert "⚠️ 수집 오류" in md and "arxiv" in md

    data = json.loads((root / "data" / "2026" / "2026-10-23.json").read_text())
    (item,) = data["items"]
    assert item["verifications"][0]["verdict"] == Verdict.VERIFIED
    assert item["verifications"][0]["evidence"][0]["revision"] == "abc1234def"
    assert data["ledger"][0]["as_expected"] is True

    seen = json.loads((root / "state" / "seen.json").read_text())
    assert seen == {"arxiv:2610.00001": "2026-10-23"}
    readme = (root / "README.md").read_text()
    assert "digests/2026/2026-10-23.md" in readme and "old" not in readme


def test_daily_runs_once_per_day_unless_forced(tmp_path, web):
    root = make_repo(tmp_path)
    stub_sources(web)
    assert run_daily(root, RUN) is not None
    assert run_daily(root, RUN) is None
    assert run_daily(root, RUN, force=True) is not None
    # 재실행해도 오늘 실린 항목은 유지된다
    data = json.loads((root / "data" / "2026" / "2026-10-23.json").read_text())
    assert len(data["items"]) == 1


def test_next_day_skips_items_already_published(tmp_path, web):
    root = make_repo(tmp_path)
    stub_sources(web)
    run_daily(root, RUN)
    web.add(
        f"{HUB}/api/daily_papers?date=2026-10-23&limit=100",
        json.loads(web.routes[f"{HUB}/api/daily_papers?date=2026-10-22&limit=100"][1]),
    )
    run_daily(root, date(2026, 10, 24))
    data = json.loads((root / "data" / "2026" / "2026-10-24.json").read_text())
    assert data["items"] == []
