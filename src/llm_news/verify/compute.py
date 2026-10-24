"""Tier A: 대규모 훈련/추론 수치 주장을 재훈련 없이 산술로 검증한다.

- training_flops: 밀집 트랜스포머 훈련 연산량 C ≈ 6·N·D (Kaplan et al. 2020, Hoffmann et al. 2022)
- kv_cache_bytes: KV 캐시 크기 = 2·L·n_kv·d_head·T·B·bytes
N을 repo로 주면 Hub의 safetensors 실측 파라미터 수를 쓴다(Tier B 근거와 결합).
"""

from __future__ import annotations

from ..models import Claim, Evidence, Tier, Verdict, Verification
from ..units import human, parse_quantity, rounding_tolerance
from . import hf
from .registry import register


def _within(claimed: float, actual: float, tol: float) -> tuple[bool, float]:
    err = abs(claimed - actual) / actual
    return err <= tol, err


@register("training_flops")
def verify_training_flops(claim: Claim) -> Verification:
    """params: flops(주장), tokens(D), params(N) 또는 repo, tolerance(기본 max(20%, 반올림)).

    6ND는 어텐션 연산을 뺀 근사이므로 기본 허용 오차를 20%로 둔다.
    MoE는 활성 파라미터를 N으로 줘야 한다.
    """
    p = claim.params
    ev: list[Evidence] = []
    if "params" in p:
        n = parse_quantity(p["params"])
        n_src = "주장된 N"
    else:
        info = hf.model_info(p["repo"])
        n = ((info or {}).get("safetensors") or {}).get("total")
        if not n:
            return Verification(claim, Verdict.UNVERIFIABLE, Tier.SOURCE, "6ND", "N을 Hub에서 확인할 수 없음")
        n_src = "safetensors 실측 N"
        ev.append(
            Evidence(
                "N 출처", url=f"{hf.HUB}/api/models/{p['repo']}", revision=info.get("sha"), data={"N": n}
            )
        )
    d = parse_quantity(p["tokens"])
    claimed = parse_quantity(p["flops"])
    actual = 6 * n * d
    tol = float(p.get("tolerance", max(0.20, rounding_tolerance(p["flops"]))))
    ok, err = _within(claimed, actual, tol)
    ev.append(Evidence("6ND 계산", data={"N": n, "D": d, "6ND": actual}))
    return Verification(
        claim,
        Verdict.VERIFIED if ok else Verdict.REFUTED,
        Tier.INTERNAL,
        "6ND",
        f"6·N·D = 6×{human(n)}({n_src})×{human(d)} = {actual:.3g} FLOPs vs 주장 {claimed:.3g} "
        f"(오차 {err:.1%}, 허용 {tol:.0%})",
        ev,
    )


@register("kv_cache_bytes")
def verify_kv_cache(claim: Claim) -> Verification:
    """params: bytes(주장), layers, kv_heads, head_dim, seq_len, batch(1), bytes_per_elem(2)."""
    p = claim.params
    layers, kv, hd = int(p["layers"]), int(p["kv_heads"]), int(p["head_dim"])
    seq, batch = int(parse_quantity(p["seq_len"])), int(p.get("batch", 1))
    bpe = float(p.get("bytes_per_elem", 2))
    actual = 2 * layers * kv * hd * seq * batch * bpe
    claimed = parse_quantity(p["bytes"])
    # GB/GiB 혼용(약 7.4%)을 흡수할 만큼의 허용 오차
    tol = float(p.get("tolerance", max(0.08, rounding_tolerance(p["bytes"]))))
    ok, err = _within(claimed, actual, tol)
    return Verification(
        claim,
        Verdict.VERIFIED if ok else Verdict.REFUTED,
        Tier.INTERNAL,
        "kv-cache-formula",
        f"2·{layers}·{kv}·{hd}·{seq}·{batch}·{bpe:g} = {human(actual)}B vs 주장 {human(claimed)}B "
        f"(오차 {err:.1%})",
        [Evidence("KV 캐시 공식", data={"bytes": actual})],
    )
