"""큐레이션·검증 파이프라인의 공통 데이터 모델.

흐름: Item(소식) -> Claim(검증 가능한 원자적 주장) -> Verification(판정 + 증거).
모든 판정은 재현 가능한 증거(Evidence)를 동반해야 한다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Verdict(StrEnum):
    # 내부 재현(수학/알고리즘) 또는 원본 아티팩트와의 대조로 주장이 확인됨
    VERIFIED = "verified"
    # 같은 근거로 주장이 틀렸음이 확인됨 (반례 또는 원본과 불일치)
    REFUTED = "refuted"
    # 원본 출처(모델 카드 등)가 스스로 같은 내용을 선언함 — 독립 검증은 아님
    ATTESTED = "attested"
    # 근거는 찾았지만 결론을 내리기에 부족함
    INCONCLUSIVE = "inconclusive"
    # 검증할 근거(아티팩트/명세)가 없음
    UNVERIFIABLE = "unverifiable"


class Tier(StrEnum):
    """검증 비용 계층. METHODOLOGY.md 참고."""

    # A: 저장소 내부에서 결정론적으로 재현 (기호 계산, 수치 재현, 알고리즘 차분 테스트)
    INTERNAL = "internal"
    # B: 대규모 훈련 결과는 재훈련하지 않고 원본 아티팩트(HF Hub 등)와 대조
    SOURCE = "source"
    # C: 자기보고 수치 등 검증 수단이 없는 주장
    REPORTED = "reported"


@dataclass
class Evidence:
    description: str
    url: str | None = None
    # 원본 저장소 커밋 sha 등, 재검증 시 같은 데이터를 보도록 고정하는 값
    revision: str | None = None
    retrieved_at: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Claim:
    kind: str
    statement: str
    params: dict[str, Any] = field(default_factory=dict)
    # 주장을 추출한 원문 구절
    quote: str | None = None
    id: str | None = None

    def __post_init__(self) -> None:
        if self.id is None:
            key = f"{self.kind}|{self.statement}|{sorted(self.params.items())!r}"
            self.id = hashlib.sha1(key.encode()).hexdigest()[:12]


@dataclass
class Verification:
    claim: Claim
    verdict: Verdict
    tier: Tier
    method: str
    summary: str
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class Item:
    source: str
    # 다이제스트 섹션: paper | model | blog
    section: str
    # 소스와 무관한 정규 식별자. 같은 논문이 HF와 arXiv 양쪽에서 와도 "arxiv:<id>"로 같다.
    id: str
    title: str
    url: str
    published: str
    summary: str = ""
    authors: list[str] = field(default_factory=list)
    # HF 업보트, 좋아요 등 소스별 인기 신호
    signals: dict[str, float] = field(default_factory=dict)
    # 이 소식과 연결된 HF 저장소 (예: {"models": [...], "datasets": [...]})
    artifacts: dict[str, list[str]] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    score: float = 0.0
    claims: list[Claim] = field(default_factory=list)
    verifications: list[Verification] = field(default_factory=list)
