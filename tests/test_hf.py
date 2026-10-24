from conftest import HUB, add_model, fixture_json, safetensors_blob

from llm_news.models import Claim, Verdict
from llm_news.verify import verify
from llm_news.verify.hf import DATASETS_SERVER, card_token_counts

QWEN = "Qwen/Qwen2.5-0.5B"


def params_claim(repo, value, scope="total"):
    return Claim("hf_model_params", f"{repo} {value}", {"repo": repo, "value": value, "scope": scope})


def test_params_verified_against_stored_and_derived(web):
    sha = add_model(web, QWEN, total=494_032_768, config=fixture_json("qwen2.5-0.5b.config.json"))
    v = verify(params_claim(QWEN, "0.5B"))
    assert v.verdict == Verdict.VERIFIED
    assert all(e.revision == sha for e in v.evidence)
    assert "차이 0.00%" in v.summary  # 아키텍처 계산과 실측이 정확히 일치


def test_params_refuted_when_name_overstates(web):
    add_model(web, QWEN, total=494_032_768, config=fixture_json("qwen2.5-0.5b.config.json"))
    assert verify(params_claim(QWEN, "1.5B")).verdict == Verdict.REFUTED


def test_non_embedding_match_is_inconclusive(web):
    # 총 494M, 임베딩 제외 358M: '0.36B'는 집계 기준 차이일 수 있으므로 반박하지 않는다
    add_model(web, QWEN, total=494_032_768, config=fixture_json("qwen2.5-0.5b.config.json"))
    v = verify(params_claim(QWEN, "0.36B"))
    assert v.verdict == Verdict.INCONCLUSIVE
    assert "임베딩 제외" in v.summary


def test_active_params_from_config(web):
    repo = "Qwen/Qwen3-30B-A3B"
    add_model(web, repo, total=30_532_122_624, config=fixture_json("qwen3-30b-a3b.config.json"))
    assert verify(params_claim(repo, "3B", "active")).verdict == Verdict.VERIFIED
    assert verify(params_claim(repo, "10B", "active")).verdict == Verdict.REFUTED


def test_active_params_from_safetensors_headers_for_unknown_architecture(web):
    """계산식이 없는 아키텍처: 텐서 헤더만 읽어 전문가 파라미터를 센다."""
    repo, sha = "lab/Toy-MoE-A1B", "deadbeef"
    shapes = {"model.embed_tokens.weight": [1000, 100]}
    for layer in range(2):
        for e in range(4):  # 전문가 4개, 각 3×100×200
            for proj in ("gate_proj", "up_proj", "down_proj"):
                shapes[f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"] = [200, 100]
        shapes[f"model.layers.{layer}.mlp.shared_experts.up_proj.weight"] = [50, 100]
    add_model(
        web,
        repo,
        sha=sha,
        config={"model_type": "toy_moe", "num_experts_per_tok": 1},
        siblings=["model.safetensors"],
    )
    web.add(f"{HUB}/{repo}/resolve/{sha}/model.safetensors", safetensors_blob(shapes))
    expert = 2 * 4 * 3 * 200 * 100
    total = 1000 * 100 + expert + 2 * 50 * 100
    active = total - expert + expert // 4
    v = verify(params_claim(repo, str(active), "active"))
    assert v.verdict == Verdict.VERIFIED
    data = next(e.data for e in v.evidence if "텐서 shape" in e.description)
    assert data["total"] == total and data["active"] == active and data["n_experts"] == 4
    # Range 요청만 썼는지 (가중치 본문 미다운로드)
    assert all("Range" in h for u, h in web.requests if u.endswith(".safetensors"))


def test_missing_repo_is_unverifiable(web):
    assert verify(params_claim("nobody/none", "7B")).verdict == Verdict.UNVERIFIABLE


def test_context_claims(web):
    add_model(web, QWEN, total=1, config=fixture_json("qwen2.5-0.5b.config.json"))
    ctx = lambda v: verify(Claim("hf_model_context", "ctx", {"repo": QWEN, "value": v})).verdict  # noqa: E731
    assert ctx("32K") == Verdict.VERIFIED
    assert ctx("8K") == Verdict.VERIFIED  # 지원 길이의 하한으로 성립
    assert ctx("1M") == Verdict.REFUTED


def test_training_tokens_attested_or_refuted_by_card(web):
    add_model(web, QWEN, total=1, card="---\nlicense: mit\n---\nPretrained on 18T tokens of text.")
    tok = lambda v: verify(Claim("hf_training_tokens", "t", {"repo": QWEN, "value": v})).verdict  # noqa: E731
    assert tok("18T") == Verdict.ATTESTED
    assert tok("36T") == Verdict.REFUTED


def test_card_token_counts():
    assert card_token_counts("trained on 11.2T tokens and 300B high-quality tokens") == [11.2e12, 300e9]


def test_dataset_rows_and_token_lower_bound(web):
    repo = "org/data"
    web.add(f"{HUB}/api/datasets/{repo}", {"id": repo, "sha": "ds1", "cardData": {"license": "mit"}})
    web.add(
        f"{DATASETS_SERVER}/size?dataset=org%2Fdata",
        {
            "size": {"dataset": {"num_rows": 52_002, "num_bytes_memory": 40_000_000}, "configs": []},
            "partial": False,
        },
    )
    ds = lambda **p: verify(Claim("hf_dataset", "d", {"repo": repo, **p})).verdict  # noqa: E731
    assert ds() == Verdict.VERIFIED
    assert ds(rows="52K") == Verdict.VERIFIED
    assert ds(rows="100K") == Verdict.REFUTED
    assert ds(tokens="10M") == Verdict.INCONCLUSIVE  # 4바이트/토큰: 모순 없음
    assert ds(tokens="1B") == Verdict.REFUTED  # 40MB에 10억 토큰은 불가능


def test_active_params_counted_without_input_embedding_is_inconclusive(web):
    """'활성 파라미터는 입력 임베딩 제외' 관행의 주장은 반박하지 않고 집계 기준 차이로 표시한다."""
    repo, sha = "lab/Toy-MoE", "cafe"
    shapes = {"model.embed_tokens.weight": [50_000, 100], "lm_head.weight": [50_000, 100]}
    for e in range(8):
        shapes[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = [1000, 100]
    add_model(
        web,
        repo,
        sha=sha,
        config={"model_type": "toy", "num_experts_per_tok": 2},
        siblings=["model.safetensors"],
    )
    web.add(f"{HUB}/{repo}/resolve/{sha}/model.safetensors", safetensors_blob(shapes))
    active = 2 * 50_000 * 100 + 2 * 1000 * 100
    assert verify(params_claim(repo, str(active), "active")).verdict == Verdict.VERIFIED
    v = verify(params_claim(repo, str(active - 50_000 * 100), "active"))
    assert v.verdict == Verdict.INCONCLUSIVE
    assert "입력 임베딩 제외" in v.summary
