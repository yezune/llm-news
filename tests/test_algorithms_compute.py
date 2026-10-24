import pytest
from conftest import ROOT

from llm_news.models import Claim, Verdict
from llm_news.verify import verify
from llm_news.verify.algorithms import ALGORITHMS, CheckResult, algorithm, load_reproductions

BUILTIN = [
    "online_softmax_attention",
    "rope_relative_position",
    "speculative_sampling_exact",
    "gqa_equals_repeated_mha",
    "lora_merge_equivalence",
    "kv_cache_incremental_decoding",
]


@pytest.mark.parametrize("name", BUILTIN)
def test_builtin_reproductions_hold(name):
    v = verify(Claim("algorithm", name, {"name": name, "trials": 5}))
    assert v.verdict == Verdict.VERIFIED, v.summary


def test_expect_false_turns_holding_property_into_refuted():
    v = verify(Claim("algorithm", "x", {"name": "gqa_equals_repeated_mha", "expect": False, "trials": 2}))
    assert v.verdict == Verdict.REFUTED


def test_broken_algorithm_is_refuted_with_failing_trial():
    @algorithm("test_broken_softmax")
    def broken(rng):
        """정규화를 빼먹은 소프트맥스는 확률 분포가 아니다."""
        x = rng.standard_normal(10)
        s = (2.718281828**x).sum()
        return CheckResult(abs(s - 1) < 1e-9, abs(s - 1))

    v = verify(Claim("algorithm", "broken", {"name": "test_broken_softmax"}))
    assert v.verdict == Verdict.REFUTED
    assert v.evidence[0].data["failed_trial"] == 0
    ALGORITHMS.pop("test_broken_softmax")


def test_unknown_reproduction_is_unverifiable():
    v = verify(Claim("algorithm", "x", {"name": "does_not_exist"}))
    assert v.verdict == Verdict.UNVERIFIABLE


def test_repository_reproductions_load():
    loaded = load_reproductions(ROOT / "reproductions")
    assert "mla_absorption" in loaded
    assert verify(Claim("algorithm", "mla", {"name": "mla_weight_absorption"})).verdict == Verdict.VERIFIED
    naive = verify(Claim("algorithm", "naive", {"name": "speculative_resample_from_target"}))
    assert naive.verdict == Verdict.REFUTED


def test_training_flops_6nd():
    ok = verify(Claim("training_flops", "llama3", {"params": "405B", "tokens": "15.6T", "flops": "3.8e25"}))
    bad = verify(Claim("training_flops", "x10", {"params": "405B", "tokens": "15.6T", "flops": "3.8e26"}))
    assert ok.verdict == Verdict.VERIFIED
    assert bad.verdict == Verdict.REFUTED


def test_kv_cache_formula():
    p = {"layers": 80, "kv_heads": 8, "head_dim": 128, "seq_len": 4096, "bytes": "1.34e9"}
    assert verify(Claim("kv_cache_bytes", "kv", p)).verdict == Verdict.VERIFIED
    assert verify(Claim("kv_cache_bytes", "kv", {**p, "bytes": "10.7e9"})).verdict == Verdict.REFUTED


def test_unregistered_kind_is_reported_tier():
    v = verify(Claim("benchmark_score", "MMLU 90%", {"value": 90}))
    assert v.verdict == Verdict.UNVERIFIABLE
    assert v.tier.value == "reported"
