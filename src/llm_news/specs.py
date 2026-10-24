"""claims/*.yaml 검증 명세 로더.

자동 추출이 못 하는 주장(수학 정리, 알고리즘 성질, 정확한 데이터셋 크기 등)을 사람이 명세로 적는다.

  title: FlashAttention
  source: https://arxiv.org/abs/2205.14135
  item: arxiv:2205.14135          # 선택: 다이제스트의 소식과 연결
  claims:
    - kind: algorithm
      statement: 블록 단위 온라인 소프트맥스는 정확한 어텐션과 같다
      params: {name: online_softmax_attention, block: 16}
      expected: verified           # 선택: 회귀 감지용 기대 판정
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import Claim, Verdict, Verification
from .verify import verify


@dataclass
class SpecResult:
    file: str
    title: str
    source: str | None
    item: str | None
    verification: Verification
    expected: Verdict | None

    @property
    def as_expected(self) -> bool:
        return self.expected is None or self.expected == self.verification.verdict


def load_spec(path: Path) -> tuple[dict[str, Any], list[tuple[Claim, Verdict | None]]]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = []
    for i, c in enumerate(doc.get("claims", [])):
        for key in ("kind", "statement"):
            if key not in c:
                raise ValueError(f"{path}: claims[{i}]에 '{key}' 없음")
        claim = Claim(c["kind"], c["statement"], c.get("params") or {}, quote=c.get("quote"), id=c.get("id"))
        expected = Verdict(c["expected"]) if c.get("expected") else None
        out.append((claim, expected))
    return doc, out


def run_specs(directory: Path) -> list[SpecResult]:
    results = []
    for path in sorted(directory.glob("**/*.y*ml")):
        doc, claims = load_spec(path)
        for claim, expected in claims:
            results.append(
                SpecResult(
                    file=str(path.relative_to(directory.parent)),
                    title=doc.get("title", path.stem),
                    source=doc.get("source"),
                    item=doc.get("item"),
                    verification=verify(claim),
                    expected=expected,
                )
            )
    return results
