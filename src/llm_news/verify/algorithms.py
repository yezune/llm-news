"""Tier A: 알고리즘 주장을 소규모 실행 재현으로 검증한다.

방법: 논문이 기술한 알고리즘을 작게 구현하고, 정의대로 계산하는 기준 구현(naive reference)과
고정 시드의 무작위 입력에서 비교(differential testing)한다. 대규모 훈련이 필요 없는
"이 알고리즘은 X와 같다/X를 보존한다" 류의 주장이 대상이다.

새 재현은 저장소 루트의 reproductions/*.py 에 @algorithm("이름") 함수로 추가한다.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..models import Claim, Evidence, Tier, Verdict, Verification
from .registry import register


@dataclass
class CheckResult:
    holds: bool
    max_error: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


Check = Callable[..., CheckResult]
ALGORITHMS: dict[str, tuple[Check, str]] = {}


def algorithm(name: str, description: str = "") -> Callable[[Check], Check]:
    def deco(fn: Check) -> Check:
        ALGORITHMS[name] = (fn, description or (fn.__doc__ or "").strip().splitlines()[0])
        return fn

    return deco


def load_reproductions(directory: Path) -> list[str]:
    """reproductions/*.py 를 import해 @algorithm 등록을 실행한다. 저장소에 커밋된(리뷰된) 코드만 대상."""
    loaded = []
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"reproductions.{path.stem}", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(path.stem)
    return loaded


@register("algorithm")
def verify_algorithm(claim: Claim) -> Verification:
    """params: name(등록된 재현 이름), expect(기본 True), seed, trials, 그 외 재현별 인자."""
    p = dict(claim.params)
    name = p.pop("name")
    expect = bool(p.pop("expect", True))
    seed = int(p.pop("seed", 0))
    trials = int(p.pop("trials", 20))
    if name not in ALGORITHMS:
        return Verification(
            claim,
            Verdict.UNVERIFIABLE,
            Tier.INTERNAL,
            "reproduction",
            f"등록된 재현 '{name}' 없음 — reproductions/에 추가 필요",
        )
    fn, desc = ALGORITHMS[name]
    worst = CheckResult(True)
    for t in range(trials):
        rng = np.random.default_rng([seed, t])
        res = fn(rng, **p)
        if not res.holds:
            worst = res
            worst.details["failed_trial"] = t
            break
        if res.max_error >= worst.max_error:
            worst = res
    ev = [
        Evidence(
            f"재현 '{name}': {desc}",
            data={"seed": seed, "trials": trials, "max_error": worst.max_error, **worst.details},
        )
    ]
    matches = worst.holds == expect
    verdict = Verdict.VERIFIED if matches else Verdict.REFUTED
    state = "성립" if worst.holds else "불성립"
    return Verification(
        claim,
        verdict,
        Tier.INTERNAL,
        f"reproduction:{name}",
        f"{trials}회 시드 고정 시행에서 성질 {state} (최대 오차 {worst.max_error:.2e}), "
        f"주장={'성립' if expect else '불성립'}",
        ev,
    )


# ---------------------------------------------------------------------------
# 기본 재현 모음 — LLM 논문에서 자주 나오는 알고리즘 주장
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _naive_attention(q, k, v, causal=False):
    s = q @ k.T / np.sqrt(q.shape[-1])
    if causal:
        s = np.where(np.tril(np.ones_like(s, dtype=bool)), s, -np.inf)
    return _softmax(s) @ v


TOL = 1e-10


@algorithm("online_softmax_attention")
def online_softmax_attention(rng, seq_len=64, dim=16, block=8, causal=True, scale=1.0):
    """FlashAttention식 블록 단위 온라인 소프트맥스가 정확한 어텐션과 같다."""
    q, k, v = (rng.standard_normal((seq_len, dim)) * scale for _ in range(3))
    out = np.zeros_like(q)
    for i0 in range(0, seq_len, block):
        qi = q[i0 : i0 + block]
        m = np.full(len(qi), -np.inf)
        denom = np.zeros(len(qi))
        acc = np.zeros_like(qi)
        for j0 in range(0, seq_len, block):
            s = qi @ k[j0 : j0 + block].T / np.sqrt(dim)
            if causal:
                rows = np.arange(i0, i0 + len(qi))[:, None]
                cols = np.arange(j0, j0 + s.shape[1])[None, :]
                s = np.where(cols <= rows, s, -np.inf)
            m_new = np.maximum(m, s.max(axis=1))
            if np.all(np.isneginf(m_new)):
                continue
            alpha = np.exp(m - m_new)
            p = np.exp(s - m_new[:, None])
            denom = denom * alpha + p.sum(axis=1)
            acc = acc * alpha[:, None] + p @ v[j0 : j0 + block]
            m = m_new
        out[i0 : i0 + block] = acc / denom[:, None]
    err = float(np.abs(out - _naive_attention(q, k, v, causal)).max())
    return CheckResult(err < TOL * max(1.0, scale**2) * 100, err)


def _rope(x: np.ndarray, pos: int, base: float) -> np.ndarray:
    d = x.shape[-1]
    inv = base ** (-np.arange(0, d, 2) / d)
    ang = pos * inv
    x1, x2 = x[0::2], x[1::2]
    out = np.empty_like(x)
    out[0::2] = x1 * np.cos(ang) - x2 * np.sin(ang)
    out[1::2] = x1 * np.sin(ang) + x2 * np.cos(ang)
    return out


@algorithm("rope_relative_position")
def rope_relative_position(rng, dim=32, base=10000.0, max_pos=4096):
    """RoPE 적용 후 q·k 내적은 절대 위치가 아니라 상대 위치(m-n)에만 의존한다."""
    q, k = rng.standard_normal(dim), rng.standard_normal(dim)
    m, n = rng.integers(0, max_pos, size=2)
    shift = int(rng.integers(0, max_pos))
    a = _rope(q, m, base) @ _rope(k, n, base)
    b = _rope(q, m + shift, base) @ _rope(k, n + shift, base)
    err = float(abs(a - b))
    return CheckResult(err < 1e-8 * max(1.0, abs(a)), err, {"m": int(m), "n": int(n), "shift": shift})


@algorithm("speculative_sampling_exact")
def speculative_sampling_exact(rng, vocab=50, draft_quality=0.5):
    """추측 디코딩의 거절 샘플링은 목표 모델 분포 p를 정확히 보존한다 (Leviathan et al. 2023).

    표본 추출이 아니라 출력 분포를 해석적으로 계산해 비교한다:
    P(x) = q(x)·min(1, p(x)/q(x)) + (1-β)·norm(max(0, p-q))(x),  β = Σ min(p, q)
    """
    p = rng.dirichlet(np.ones(vocab))
    noise = rng.dirichlet(np.ones(vocab))
    q = draft_quality * p + (1 - draft_quality) * noise
    accept = np.minimum(p, q)
    beta = accept.sum()
    residual = np.maximum(p - q, 0)
    out = accept + (1 - beta) * (residual / residual.sum() if residual.sum() > 0 else 0)
    err = float(np.abs(out - p).max())
    return CheckResult(err < 1e-12, err, {"acceptance_rate": float(beta)})


@algorithm("gqa_equals_repeated_mha")
def gqa_equals_repeated_mha(rng, seq_len=16, head_dim=8, n_heads=8, n_kv_heads=2):
    """GQA는 KV 헤드를 그룹 수만큼 복제한 MHA와 출력이 같다 (Ainslie et al. 2023)."""
    assert n_heads % n_kv_heads == 0
    g = n_heads // n_kv_heads
    q = rng.standard_normal((n_heads, seq_len, head_dim))
    k = rng.standard_normal((n_kv_heads, seq_len, head_dim))
    v = rng.standard_normal((n_kv_heads, seq_len, head_dim))
    gqa = np.stack([_naive_attention(q[h], k[h // g], v[h // g], True) for h in range(n_heads)])
    k_rep, v_rep = np.repeat(k, g, axis=0), np.repeat(v, g, axis=0)
    mha = np.stack([_naive_attention(q[h], k_rep[h], v_rep[h], True) for h in range(n_heads)])
    err = float(np.abs(gqa - mha).max())
    return CheckResult(err < TOL, err)


@algorithm("lora_merge_equivalence")
def lora_merge_equivalence(rng, d_in=64, d_out=48, rank=4, alpha=8.0, n=10):
    """LoRA 가중치 병합 W' = W + (α/r)·BA 는 어댑터 경로 출력과 같고, 추가 파라미터는 r·(d_in+d_out)."""
    W = rng.standard_normal((d_out, d_in))
    A = rng.standard_normal((rank, d_in))
    B = rng.standard_normal((d_out, rank))
    x = rng.standard_normal((n, d_in))
    s = alpha / rank
    adapter = x @ W.T + s * (x @ A.T) @ B.T
    merged = x @ (W + s * B @ A).T
    err = float(np.abs(adapter - merged).max())
    extra = A.size + B.size
    ok = err < 1e-9 and extra == rank * (d_in + d_out)
    return CheckResult(ok, err, {"extra_params": extra})


@algorithm("kv_cache_incremental_decoding")
def kv_cache_incremental_decoding(rng, seq_len=24, dim=16):
    """KV 캐시로 한 토큰씩 디코딩한 결과는 매번 전체 인과 어텐션을 다시 계산한 결과와 같다."""
    q, k, v = (rng.standard_normal((seq_len, dim)) for _ in range(3))
    full = _naive_attention(q, k, v, causal=True)
    k_cache, v_cache, inc = [], [], []
    for t in range(seq_len):
        k_cache.append(k[t])
        v_cache.append(v[t])
        inc.append(_naive_attention(q[t : t + 1], np.array(k_cache), np.array(v_cache))[0])
    err = float(np.abs(np.array(inc) - full).max())
    return CheckResult(err < TOL, err)
