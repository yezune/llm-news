"""Tier A: 수학 주장을 저장소 내부에서 검증한다.

판정 원칙:
- VERIFIED는 기호적 증명(sympy가 lhs-rhs를 0으로 정리)일 때만 준다.
- 수치 반례가 하나라도 나오면 REFUTED (반례를 증거로 남김).
- 반례는 없지만 증명도 못 하면 INCONCLUSIVE. "많이 시험해서 맞았다"는 증명이 아니다.
"""

from __future__ import annotations

import random
import re
import signal
import threading
from contextlib import contextmanager
from typing import Any

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

from ..models import Claim, Evidence, Tier, Verdict, Verification
from .registry import register

# 수식에서 허용하는 sympy 이름. parse_expr는 내부적으로 eval을 쓰므로 이 밖의 식별자는 거부한다.
ALLOWED_FUNCS = {
    name: getattr(sp, name)
    for name in (
        "sin cos tan exp log sqrt Abs sign tanh sinh cosh atan asin acos erf gamma "
        "factorial binomial Sum Product Integral Derivative diff integrate limit Max Min "
        "floor ceiling pi E oo I Rational Integer Float Symbol Matrix Piecewise"
    ).split()
}
# 숫자 뒤에 붙은 지수 표기(1e3의 e3)는 식별자가 아니다.
_IDENT = re.compile(r"(?<![0-9.])[A-Za-z_][A-Za-z_0-9]*")
TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication)
ASSUMPTIONS = {"real", "positive", "nonnegative", "integer", "negative", "nonzero"}


class UnsafeExpression(ValueError):
    pass


def make_symbols(variables: dict[str, Any]) -> dict[str, sp.Symbol]:
    """variables: {"x": "positive"} 또는 {"n": ["positive", "integer"]}."""
    out = {}
    for name, assum in (variables or {}).items():
        flags = [assum] if isinstance(assum, str) else list(assum or ["real"])
        bad = set(flags) - ASSUMPTIONS
        if bad:
            raise ValueError(f"알 수 없는 가정 {bad}")
        out[name] = sp.Symbol(name, **{f: True for f in flags})
    return out


def safe_parse(text: str, symbols: dict[str, sp.Symbol]) -> sp.Expr:
    if "__" in text or re.search(r"[A-Za-z_]\s*\.", text) or any(c in text for c in ";'\"`\\"):
        raise UnsafeExpression(f"허용되지 않는 구문: {text!r}")
    for ident in _IDENT.findall(text):
        if ident not in ALLOWED_FUNCS and ident not in symbols:
            raise UnsafeExpression(f"선언되지 않은 식별자 '{ident}' (variables에 선언 필요)")
    global_dict: dict[str, Any] = {"__builtins__": {}, **ALLOWED_FUNCS}
    return parse_expr(text, local_dict=dict(symbols), global_dict=global_dict, transformations=TRANSFORMS)


def _sample(sym: sp.Symbol, rng: random.Random, domain: list[float] | None) -> sp.Expr:
    if domain:
        lo, hi = domain
        if sym.is_integer:
            return sp.Integer(rng.randint(int(lo), int(hi)))
        return sp.Float(rng.uniform(lo, hi), 30)
    if sym.is_integer:
        lo = 1 if sym.is_positive else (0 if sym.is_nonnegative else -20)
        return sp.Integer(rng.randint(lo, 30))
    if sym.is_positive or sym.is_nonnegative:
        return sp.Float(10 ** rng.uniform(-3, 3), 30)
    if sym.is_negative:
        return sp.Float(-(10 ** rng.uniform(-3, 3)), 30)
    return sp.Float(rng.uniform(-10, 10), 30)


def _numeric(expr: sp.Expr, subs: dict[sp.Symbol, sp.Expr]) -> complex | None:
    try:
        val = complex(sp.N(expr.subs(subs), 30))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if val != val or abs(val) == float("inf"):  # NaN/inf: 정의역 밖 표본
        return None
    return val


def find_counterexample(
    check, symbols: dict[str, sp.Symbol], domains: dict[str, list[float]], trials: int, seed: int
) -> tuple[dict[str, str] | None, int]:
    """check(subs) -> True(성립)/False(위반)/None(평가 불가). 결정론적(seed 고정)."""
    rng = random.Random(seed)
    tested = 0
    for _ in range(trials):
        subs = {s: _sample(s, rng, domains.get(name)) for name, s in symbols.items()}
        ok = check(subs)
        if ok is None:
            continue
        tested += 1
        if not ok:
            return {str(k): str(v) for k, v in subs.items()}, tested
    return None, tested


SIMPLIFY_SECONDS = 20


class ProofTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds: int):
    """기호 계산이 끝나지 않는 경우를 막는다. 메인 스레드(Unix)에서만 동작하고, 그 밖에서는 제한 없음."""
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise(signum, frame):
        raise ProofTimeout(f"{seconds}초 안에 기호 계산이 끝나지 않음")

    old = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _interval(sym: sp.Symbol, domain: list[float] | None) -> sp.Set:
    if domain:
        return sp.Interval(domain[0], domain[1])
    if sym.is_positive:
        return sp.Interval.open(0, sp.oo)
    if sym.is_nonnegative:
        return sp.Interval(0, sp.oo)
    if sym.is_negative:
        return sp.Interval.open(-sp.oo, 0)
    return sp.S.Reals


def prove_univariate(gap: sp.Expr, sym: sp.Symbol, domain: list[float] | None, strict: bool) -> str | None:
    """gap >= 0 (strict면 > 0)을 정의역 위 최솟값(하한)으로 증명. 성공하면 하한을 문자열로 반환."""
    from sympy.calculus.util import minimum

    try:
        with time_limit(SIMPLIFY_SECONDS):
            low = minimum(gap, sym, _interval(sym, domain))
    except (ProofTimeout, NotImplementedError, ValueError, TypeError):
        return None
    if low is None or not low.is_comparable:
        return None
    ok = (low > 0) if strict else (low >= 0)
    return str(low) if ok else None


def _parse_common(claim: Claim):
    p = claim.params
    symbols = make_symbols(p.get("variables", {}))
    return p, symbols, p.get("domain", {}), int(p.get("trials", 200)), int(p.get("seed", 0))


def _decide(
    claim: Claim,
    proof: tuple[str, str] | None,
    check,
    symbols,
    domains,
    trials,
    seed,
    ev: list[Evidence],
) -> Verification:
    """기호 증명과 수치 반례 탐색을 둘 다 수행하고 종합한다.

    증명이 있어도 반례 탐색을 돌리는 이유: sympy 자체의 버그로 잘못된 '증명'이 나오는 경우를 잡기 위해서다.
    """
    cex, tested = find_counterexample(check, symbols, domains, trials, seed)
    if cex is not None:
        ev.append(Evidence("수치 반례", data={"point": cex, "seed": seed}))
        if proof:
            return Verification(
                claim,
                Verdict.INCONCLUSIVE,
                Tier.INTERNAL,
                proof[0],
                f"기호 증명({proof[1]})과 수치 반례 {cex}가 충돌 — 수동 확인 필요",
                ev,
            )
        return Verification(
            claim, Verdict.REFUTED, Tier.INTERNAL, "numeric-counterexample", f"반례 발견: {cex}", ev
        )
    ev.append(Evidence("무작위 표본 검사", data={"tested": tested, "seed": seed}))
    if proof:
        return Verification(
            claim,
            Verdict.VERIFIED,
            Tier.INTERNAL,
            proof[0],
            f"{proof[1]}; 교차 확인으로 {tested}개 표본에서 반례 없음",
            ev,
        )
    return Verification(
        claim,
        Verdict.INCONCLUSIVE,
        Tier.INTERNAL,
        "numeric-sampling",
        f"기호적 증명 실패, {tested}개 표본에서 반례 없음",
        ev,
    )


@register("math_identity")
def verify_identity(claim: Claim) -> Verification:
    """params: lhs, rhs, variables, domain(선택), trials, seed, rel_tol."""
    p, symbols, domains, trials, seed = _parse_common(claim)
    lhs = safe_parse(p["lhs"], symbols)
    rhs = safe_parse(p["rhs"], symbols)
    rel_tol = float(p.get("rel_tol", 1e-12))
    proof = None
    try:
        with time_limit(SIMPLIFY_SECONDS):
            diff = sp.simplify((lhs - rhs).doit())
    except ProofTimeout as exc:
        diff = str(exc)
    ev = [Evidence("sympy 정리 결과 (lhs - rhs)", data={"simplified": str(diff)})]
    if diff == 0:
        proof = ("sympy.simplify", "기호적으로 lhs - rhs = 0 증명")

    def check(subs):
        a, b = _numeric(lhs, subs), _numeric(rhs, subs)
        if a is None or b is None:
            return None
        return abs(a - b) <= rel_tol * max(1.0, abs(a), abs(b))

    return _decide(claim, proof, check, symbols, domains, trials, seed, ev)


_OPS = {"<=": sp.Le, "<": sp.Lt, ">=": sp.Ge, ">": sp.Gt}


@register("math_inequality")
def verify_inequality(claim: Claim) -> Verification:
    """params: lhs, op('<=' 등), rhs, variables, domain, trials, seed.

    증명 경로: (1) sympy가 변수 가정만으로 관계식을 참으로 정리, 또는
    (2) 단변수일 때 정의역 위에서 양변 차의 최솟값(하한)을 기호적으로 구해 부호 확인.
    """
    p, symbols, domains, trials, seed = _parse_common(claim)
    op = p.get("op", "<=")
    if op not in _OPS:
        raise ValueError(f"지원하지 않는 연산자 {op}")
    lhs = safe_parse(p["lhs"], symbols)
    rhs = safe_parse(p["rhs"], symbols)
    proof = None
    try:
        with time_limit(SIMPLIFY_SECONDS):
            rel = _OPS[op](sp.simplify(lhs - rhs), 0)
    except ProofTimeout as exc:
        rel = str(exc)
    ev = [Evidence("sympy 관계식 정리", data={"relation": str(rel)})]
    if rel is sp.true:
        proof = ("sympy.relational", "변수 가정 하에서 기호적으로 참")
    free = (lhs - rhs).free_symbols
    if proof is None and len(free) == 1:
        (sym,) = free
        gap = (rhs - lhs) if op in ("<=", "<") else (lhs - rhs)
        dom = domains.get(str(sym))
        low = prove_univariate(gap, sym, dom, strict=op in ("<", ">"))
        if low is not None:
            ev.append(Evidence("정의역 위 하한", data={"gap": str(gap), "infimum": low}))
            sign = ">" if op in ("<", ">") else "≥"
            proof = ("sympy.minimum", f"정의역 {_interval(sym, dom)}에서 양변 차의 하한 {low} {sign} 0")

    def check(subs):
        a, b = _numeric(lhs, subs), _numeric(rhs, subs)
        if a is None or b is None or abs(a.imag) > 1e-20 or abs(b.imag) > 1e-20:
            return None
        return bool(_OPS[op](sp.Float(a.real, 30), sp.Float(b.real, 30)))

    return _decide(claim, proof, check, symbols, domains, trials, seed, ev)


@register("math_value")
def verify_value(claim: Claim) -> Verification:
    """닫힌 식의 값 주장. params: expr, value, rel_tol(기본 1e-6)."""
    p = claim.params
    symbols = make_symbols(p.get("variables", {}))
    expr = safe_parse(p["expr"], symbols)
    if expr.free_symbols:
        raise ValueError("math_value는 자유 변수가 없어야 함")
    actual = complex(sp.N(expr, 50))
    claimed = complex(sp.N(safe_parse(str(p["value"]), {}), 50))
    rel_tol = float(p.get("rel_tol", 1e-6))
    err = abs(actual - claimed) / max(abs(actual), 1e-300)
    ev = [Evidence("50자리 정밀도 계산", data={"actual": str(sp.N(expr, 20)), "rel_err": err})]
    verdict = Verdict.VERIFIED if err <= rel_tol else Verdict.REFUTED
    return Verification(
        claim,
        verdict,
        Tier.INTERNAL,
        "sympy.N",
        f"계산값 {sp.N(expr, 12)} vs 주장 {p['value']} (상대오차 {err:.3g})",
        ev,
    )
