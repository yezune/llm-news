import pytest

from llm_news.models import Claim, Verdict
from llm_news.verify import verify
from llm_news.verify.symbolic import UnsafeExpression, safe_parse


def identity(lhs, rhs, **variables):
    return verify(Claim("math_identity", f"{lhs}={rhs}", {"lhs": lhs, "rhs": rhs, "variables": variables}))


def test_identity_proven_symbolically():
    v = identity(
        "exp(x + c) / (exp(x + c) + exp(y + c))", "exp(x) / (exp(x) + exp(y))", x="real", y="real", c="real"
    )
    assert v.verdict == Verdict.VERIFIED
    assert v.method == "sympy.simplify"


def test_identity_refuted_with_counterexample():
    v = identity("(a + b)^2", "a^2 + b^2", a="real", b="real")
    assert v.verdict == Verdict.REFUTED
    point = next(e for e in v.evidence if e.description == "수치 반례").data["point"]
    assert set(point) == {"a", "b"}


def test_counterexample_search_is_deterministic():
    a = identity("(a + b)^2", "a^2 + b^2", a="real", b="real")
    b = identity("(a + b)^2", "a^2 + b^2", a="real", b="real")
    assert a.summary == b.summary


def test_sampling_without_proof_is_only_inconclusive(monkeypatch):
    # 기호 증명이 실패하면, 표본에서 반례가 없더라도 VERIFIED를 주지 않는다
    import llm_news.verify.symbolic as sym

    monkeypatch.setattr(sym.sp, "simplify", lambda e: e)
    v = identity("sin(x)^2 + cos(x)^2", "1", x="real")
    assert v.verdict == Verdict.INCONCLUSIVE
    assert "반례 없음" in v.summary


def test_proof_contradicted_by_counterexample_is_not_trusted(monkeypatch):
    # sympy가 (버그로) 거짓 항등식을 0으로 정리해도, 수치 반례가 있으면 VERIFIED가 아니다
    import llm_news.verify.symbolic as sym

    monkeypatch.setattr(sym.sp, "simplify", lambda e: sym.sp.Integer(0))
    v = identity("x^2", "x", x="real")
    assert v.verdict == Verdict.INCONCLUSIVE
    assert "충돌" in v.summary


def test_inequality_proven_by_infimum():
    v = verify(
        Claim(
            "math_inequality",
            "log(1+x)<=x",
            {"lhs": "log(1 + x)", "op": "<=", "rhs": "x", "variables": {"x": "positive"}},
        )
    )
    assert v.verdict == Verdict.VERIFIED
    assert v.method == "sympy.minimum"


def test_inequality_refuted():
    v = verify(
        Claim("math_inequality", "x^2>=x", {"lhs": "x^2", "op": ">=", "rhs": "x", "variables": {"x": "real"}})
    )
    assert v.verdict == Verdict.REFUTED


def test_math_value():
    ok = verify(Claim("math_value", "e^2", {"expr": "exp(2)", "value": 7.389056}))
    bad = verify(Claim("math_value", "e^2", {"expr": "exp(2)", "value": 7.5}))
    assert ok.verdict == Verdict.VERIFIED
    assert bad.verdict == Verdict.REFUTED


@pytest.mark.parametrize(
    "expr",
    ["__import__('os').system('true')", "x.__class__", "open('/etc/passwd')", "eval('1')", "y + 1"],
)
def test_safe_parse_rejects_code_and_undeclared_names(expr):
    import sympy as sp

    with pytest.raises(UnsafeExpression):
        safe_parse(expr, {"x": sp.Symbol("x")})


def test_safe_parse_allows_scientific_notation():
    import sympy as sp

    x = sp.Symbol("x")
    assert sp.simplify(safe_parse("1e3 * x", {"x": x}) - 1000 * x) == 0


def test_verifier_errors_become_inconclusive_not_crash():
    v = verify(Claim("math_identity", "bad", {"lhs": "x +", "rhs": "x", "variables": {"x": "real"}}))
    assert v.verdict == Verdict.INCONCLUSIVE
    assert "오류" in v.summary
