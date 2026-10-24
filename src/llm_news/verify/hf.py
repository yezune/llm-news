"""Tier B: 대규모 훈련 주장을 재훈련 없이 Hugging Face Hub 원본 아티팩트와 대조한다.

근거의 우선순위:
  1. safetensors 헤더에서 집계된 실제 가중치 수 (Hub API `safetensors.total`)
  2. config.json에서 독립 계산한 아키텍처 파라미터 수 (params.count_params)
  3. 모델 카드/데이터셋 카드의 자기 선언 (ATTESTED까지만 인정)
모든 증거는 조회한 저장소 커밋 sha로 고정한다.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

from .. import http
from ..models import Claim, Evidence, Tier, Verdict, Verification
from ..units import human, parse_quantity, rounding_tolerance
from .params import context_lengths, count_params
from .registry import register
from .tensors import stats_from_hub

HUB = "https://huggingface.co"
DATASETS_SERVER = "https://datasets-server.huggingface.co"
# 이름 표기('7B')의 반올림 구간과 이 값 중 큰 쪽을 허용 오차로 쓴다.
DEFAULT_REL_TOL = 0.10


def model_info(repo: str) -> dict[str, Any] | None:
    return http.try_get_json(f"{HUB}/api/models/{repo}")


def model_config(repo: str, revision: str) -> dict[str, Any] | None:
    return http.try_get_json(f"{HUB}/{repo}/resolve/{revision}/config.json")


def model_card(repo: str, revision: str) -> str | None:
    try:
        return http.get_text(f"{HUB}/{repo}/resolve/{revision}/README.md")
    except Exception:
        return None


def _missing(claim: Claim, method: str, what: str) -> Verification:
    return Verification(
        claim, Verdict.UNVERIFIABLE, Tier.SOURCE, method, f"원본 아티팩트를 찾을 수 없음: {what}"
    )


def _tolerance(claim: Claim) -> float:
    if "tolerance" in claim.params:
        return float(claim.params["tolerance"])
    return max(DEFAULT_REL_TOL, rounding_tolerance(claim.params["value"]))


def _rel_err(claimed: float, actual: float) -> float:
    """주장값 기준 상대 오차. '7B'의 반올림 구간(±0.5B)과 같은 기준으로 비교하기 위함."""
    return abs(claimed - actual) / claimed if claimed else float("inf")


@register("hf_model_params")
def verify_model_params(claim: Claim) -> Verification:
    """params: repo, value ('7B' 등), scope ('total' | 'active'), tolerance(선택)."""
    method = "hf_model_params"
    repo = claim.params["repo"]
    scope = claim.params.get("scope", "total")
    info = model_info(repo)
    if info is None:
        return _missing(claim, method, repo)
    sha = info.get("sha")
    claimed = parse_quantity(claim.params["value"])
    tol = _tolerance(claim)

    evidence: list[Evidence] = []
    stored = (info.get("safetensors") or {}).get("total")
    if stored:
        evidence.append(
            Evidence(
                "safetensors 헤더 집계 파라미터 수",
                url=f"{HUB}/api/models/{repo}",
                revision=sha,
                retrieved_at=http.now_iso(),
                data={"total": stored, "by_dtype": info["safetensors"].get("parameters")},
            )
        )

    config = model_config(repo, sha or "main")
    derived = count_params(config) if config else None
    if derived:
        evidence.append(
            Evidence(
                "config.json 아키텍처로 독립 계산한 파라미터 수",
                url=f"{HUB}/{repo}/blob/{sha}/config.json",
                revision=sha,
                data={
                    "total": derived.total,
                    "active": derived.active,
                    "non_embedding": derived.non_embedding,
                },
            )
        )

    # 아키텍처 계산식이 없거나 가중치 메타데이터가 없으면 safetensors 헤더에서 직접 센다.
    tensors = None
    if (scope == "active" and not derived) or (not stored and not derived):
        siblings = [x["rfilename"] for x in info.get("siblings", [])]
        try:
            tensors = stats_from_hub(repo, sha or "main", siblings, config)
        except (http.FetchError, ValueError) as exc:
            evidence.append(Evidence(f"safetensors 헤더 읽기 실패: {exc}"))
        if tensors:
            evidence.append(
                Evidence(
                    "safetensors 헤더의 텐서 shape로 센 파라미터 수 (가중치 본문 미다운로드)",
                    url=f"{HUB}/{repo}/tree/{sha}",
                    revision=sha,
                    data={
                        "total": tensors.total,
                        "active": tensors.active,
                        "expert_params": tensors.expert_total,
                        "embedding": tensors.embedding,
                        "n_experts": tensors.n_experts,
                        "top_k": tensors.top_k,
                        "shards": tensors.shards,
                    },
                )
            )

    if scope == "active":
        if derived:
            actual, basis = derived.active, "config 기반 활성 파라미터"
        elif tensors and tensors.active:
            actual = tensors.active
            basis = f"텐서 헤더 기반 활성 파라미터(전문가 {tensors.n_experts}개 중 {tensors.top_k}개)"
        else:
            return Verification(
                claim,
                Verdict.UNVERIFIABLE,
                Tier.SOURCE,
                method,
                "활성 파라미터를 계산할 근거 없음(미지원 아키텍처이고 top-k 또는 전문가 텐서를 찾지 못함)",
                evidence,
            )
    elif stored:
        actual, basis = stored, "safetensors 실측"
    elif tensors:
        actual, basis = tensors.total, "safetensors 헤더 집계"
    elif derived:
        actual, basis = derived.total, "config 기반 계산(가중치 메타데이터 없음)"
    else:
        return Verification(
            claim,
            Verdict.UNVERIFIABLE,
            Tier.SOURCE,
            method,
            "가중치 메타데이터와 지원되는 config가 모두 없음",
            evidence,
        )

    cross = ""
    if stored and derived and scope == "total":
        drift = _rel_err(derived.total, stored)
        cross = f"; 아키텍처 계산 {human(derived.total)}와 실측 차이 {drift:.2%}"
        if drift > 0.02:
            cross += " (계산식 미지원 구성요소 가능성)"

    err = _rel_err(claimed, actual)
    detail = f"주장 {human(claimed)} vs {basis} {human(actual)} (오차 {err:.2%}, 허용 {tol:.0%}){cross}"
    if err <= tol:
        return Verification(claim, Verdict.VERIFIED, Tier.SOURCE, method, detail, evidence)
    # 입력 임베딩을 빼고 세는 관행("활성 파라미터는 임베딩 제외")과 맞으면 반박하지 않는다.
    embedding = derived.embedding if derived else (tensors.embedding if tensors else 0)
    if embedding and _rel_err(claimed, actual - embedding) <= tol:
        return Verification(
            claim,
            Verdict.INCONCLUSIVE,
            Tier.SOURCE,
            method,
            f"{detail}; 단 입력 임베딩 제외 수 {human(actual - embedding)}와는 일치 — 집계 기준 차이",
            evidence,
        )
    return Verification(claim, Verdict.REFUTED, Tier.SOURCE, method, detail, evidence)


@register("hf_model_context")
def verify_model_context(claim: Claim) -> Verification:
    """params: repo, value ('128K' 등)."""
    method = "hf_model_context"
    repo = claim.params["repo"]
    info = model_info(repo)
    if info is None:
        return _missing(claim, method, repo)
    sha = info.get("sha") or "main"
    config = model_config(repo, sha)
    if not config:
        return _missing(claim, method, f"{repo}/config.json")
    candidates = context_lengths(config)
    claimed = parse_quantity(claim.params["value"])
    ev = [
        Evidence(
            "config.json의 위치 임베딩 길이와 RoPE 스케일링",
            url=f"{HUB}/{repo}/blob/{sha}/config.json",
            revision=sha,
            data={"candidates": candidates, "rope_scaling": config.get("rope_scaling")},
        )
    ]
    # '128K'는 128,000일 수도 131,072일 수도 있다.
    tol = max(0.05, rounding_tolerance(claim.params["value"]))
    detail = f"주장 {human(claimed)} vs config 후보 {[human(c) for c in candidates]}"
    if not candidates:
        return Verification(claim, Verdict.UNVERIFIABLE, Tier.SOURCE, method, detail, ev)
    if any(_rel_err(claimed, c) <= tol for c in candidates):
        return Verification(claim, Verdict.VERIFIED, Tier.SOURCE, method, detail + " — 일치", ev)
    # '컨텍스트 X를 지원한다'는 주장은 config가 더 긴 길이를 허용해도 성립한다.
    if claimed < max(candidates):
        return Verification(
            claim,
            Verdict.VERIFIED,
            Tier.SOURCE,
            method,
            detail + " — config가 더 긴 길이를 허용(주장은 하한으로 성립)",
            ev,
        )
    return Verification(
        claim, Verdict.REFUTED, Tier.SOURCE, method, detail + " — config보다 긴 길이를 주장", ev
    )


_TOKENS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(T|trillion|B|billion)\s+(?:high[- ]quality\s+|training\s+|pre-?training\s+)?tokens",
    re.IGNORECASE,
)


def card_token_counts(card: str) -> list[float]:
    return [parse_quantity(f"{n}{u}") for n, u in _TOKENS_RE.findall(card)]


@register("hf_training_tokens")
def verify_training_tokens(claim: Claim) -> Verification:
    """훈련 토큰 수는 가중치로 역산할 수 없다. 원본 모델 카드의 선언과 대조해 ATTESTED/REFUTED로 판정.

    params: repo, value ('15T' 등).
    """
    method = "hf_training_tokens"
    repo = claim.params["repo"]
    info = model_info(repo)
    if info is None:
        return _missing(claim, method, repo)
    sha = info.get("sha") or "main"
    card = model_card(repo, sha)
    if not card:
        return _missing(claim, method, f"{repo}/README.md")
    claimed = parse_quantity(claim.params["value"])
    declared = card_token_counts(card)
    ev = [
        Evidence(
            "모델 카드에 선언된 훈련 토큰 수",
            url=f"{HUB}/{repo}/blob/{sha}/README.md",
            revision=sha,
            data={"declared": declared},
        )
    ]
    tol = max(0.05, rounding_tolerance(claim.params["value"]))
    detail = f"주장 {human(claimed)} vs 카드 선언 {[human(d) for d in declared] or '없음'}"
    if any(_rel_err(claimed, d) <= tol for d in declared):
        return Verification(claim, Verdict.ATTESTED, Tier.SOURCE, method, detail, ev)
    if declared:
        return Verification(claim, Verdict.REFUTED, Tier.SOURCE, method, detail + " — 원본 카드와 불일치", ev)
    return Verification(claim, Verdict.UNVERIFIABLE, Tier.SOURCE, method, detail, ev)


@register("hf_dataset")
def verify_dataset(claim: Claim) -> Verification:
    """데이터셋 주장을 실제 원본 데이터 메타데이터와 대조.

    params: repo, rows(선택), tokens(선택), config(선택).
    - rows: 데이터셋 뷰어가 실제 parquet을 읽어 센 행 수와 비교 → VERIFIED/REFUTED
    - tokens: 토큰은 직접 세지 않는다. 메모리상 바이트 수로 물리적 하한만 검사한다
      (토큰 1개는 최소 1바이트 이상의 텍스트) → 하한 위반이면 REFUTED, 아니면 INCONCLUSIVE
    """
    method = "hf_dataset"
    repo = claim.params["repo"]
    info = http.try_get_json(f"{HUB}/api/datasets/{repo}")
    if info is None:
        return _missing(claim, method, f"datasets/{repo}")
    sha = info.get("sha")
    size = http.try_get_json(f"{DATASETS_SERVER}/size?dataset={quote(repo, safe='')}")
    ev = [
        Evidence(
            "데이터셋 저장소 메타데이터",
            url=f"{HUB}/datasets/{repo}",
            revision=sha,
            retrieved_at=http.now_iso(),
            data={
                "gated": info.get("gated"),
                "license": (info.get("cardData") or {}).get("license"),
                "downloads": info.get("downloads"),
            },
        )
    ]
    if not size or "size" not in size:
        return Verification(
            claim,
            Verdict.INCONCLUSIVE,
            Tier.SOURCE,
            method,
            "저장소는 존재하나 데이터셋 뷰어 크기 정보 없음",
            ev,
        )
    want_config = claim.params.get("config")
    stats = size["size"]["dataset"]
    if want_config:
        stats = next((c for c in size["size"].get("configs", []) if c["config"] == want_config), None)
        if stats is None:
            return _missing(claim, method, f"{repo} config={want_config}")
    ev.append(
        Evidence(
            "데이터셋 뷰어가 parquet에서 집계한 크기",
            url=f"{DATASETS_SERVER}/size?dataset={repo}",
            revision=sha,
            data={
                "num_rows": stats.get("num_rows"),
                "num_bytes_memory": stats.get("num_bytes_memory"),
                "num_bytes_parquet_files": stats.get("num_bytes_parquet_files"),
                "partial": size.get("partial"),
            },
        )
    )
    partial = bool(size.get("partial"))
    results: list[tuple[Verdict, str]] = []
    if "rows" in claim.params:
        claimed = parse_quantity(claim.params["rows"])
        actual = stats.get("num_rows") or 0
        tol = max(0.01, rounding_tolerance(claim.params["rows"]))
        err = _rel_err(claimed, actual)
        msg = f"행 수 주장 {human(claimed)} vs 실제 {human(actual)} (오차 {err:.2%})"
        if partial:
            results.append((Verdict.INCONCLUSIVE, msg + " — 뷰어가 일부만 집계"))
        else:
            results.append((Verdict.VERIFIED if err <= tol else Verdict.REFUTED, msg))
    if "tokens" in claim.params:
        claimed = parse_quantity(claim.params["tokens"])
        nbytes = stats.get("num_bytes_memory") or 0
        ratio = nbytes / claimed if claimed else 0
        msg = f"토큰 주장 {human(claimed)}, 데이터 {human(nbytes)}B → 토큰당 {ratio:.2f}바이트"
        if ratio < 1 and not partial:
            results.append((Verdict.REFUTED, msg + " — 데이터 크기가 주장 토큰 수를 담을 수 없음"))
        else:
            results.append((Verdict.INCONCLUSIVE, msg + " — 크기와 모순되지 않음(토큰은 직접 세지 않음)"))
    if not results:
        return Verification(
            claim, Verdict.VERIFIED, Tier.SOURCE, method, "데이터셋이 실제로 존재하고 접근 가능", ev
        )
    # 가장 약한 판정을 전체 판정으로 쓴다.
    order = [Verdict.REFUTED, Verdict.INCONCLUSIVE, Verdict.VERIFIED]
    verdict = min((v for v, _ in results), key=order.index)
    return Verification(claim, verdict, Tier.SOURCE, method, "; ".join(m for _, m in results), ev)
