"""주장 종류(kind) -> 검증기 함수 등록부."""

from __future__ import annotations

from collections.abc import Callable

from ..models import Claim, Tier, Verdict, Verification

Verifier = Callable[[Claim], Verification]
VERIFIERS: dict[str, Verifier] = {}


def register(kind: str) -> Callable[[Verifier], Verifier]:
    def deco(fn: Verifier) -> Verifier:
        if kind in VERIFIERS:
            raise ValueError(f"검증기 중복 등록: {kind}")
        VERIFIERS[kind] = fn
        return fn

    return deco


def verify(claim: Claim) -> Verification:
    fn = VERIFIERS.get(claim.kind)
    if fn is None:
        return Verification(
            claim,
            Verdict.UNVERIFIABLE,
            Tier.REPORTED,
            "none",
            f"'{claim.kind}' 종류의 검증기가 없음 — 자기보고 수치로 취급",
        )
    try:
        return fn(claim)
    except Exception as exc:  # 검증기 오류는 판정을 내리지 않은 것으로 기록
        return Verification(
            claim,
            Verdict.INCONCLUSIVE,
            Tier.INTERNAL,
            claim.kind,
            f"검증 중 오류: {type(exc).__name__}: {exc}",
        )
