"""소식 본문에서 검증 가능한 원자적 주장을 규칙 기반으로 추출하고, 원본 아티팩트에 연결한다.

규칙 기반을 쓰는 이유: 추출 자체가 결정론적이어야 같은 입력에서 같은 검증 결과가 나온다.
연결할 원본(HF 저장소)이 없는 수치는 '자기보고(reported_*)'로 남겨 검증되지 않았음을 드러낸다.
수학 주장은 자동 추출하지 않는다. 대신 math-candidate 태그를 달아 사람이 claims/에 명세를 쓰게 한다.
"""

from __future__ import annotations

import re

from .models import Claim, Item
from .units import parse_quantity

_N = r"(\d+(?:\.\d+)?)"
PARAMS_RE = re.compile(
    rf"\b{_N}\s?(B|M|T|billion|million|trillion)[- ]?(?:total\s+)?param(?:eter)?s?\b", re.I
)
ACTIVE_RE = re.compile(rf"\b{_N}\s?(B|M|billion|million)\s+(?:active|activated)\s+param", re.I)
TOKENS_RE = re.compile(
    rf"(?:train|trained|pre-?train(?:ed|ing)?|corpus|on)\D{{0,40}}?\b{_N}\s?(T|B|trillion|billion)\s+tokens\b",
    re.I,
)
CONTEXT_RE = re.compile(
    rf"\b{_N}\s?([KM])(?:-token)?\s+(?:token\s+)?context|context\s+(?:window|length)\s+of\s+(?:up\s+to\s+)?{_N}\s?([KM])\b",
    re.I,
)
FLOPS_RE = re.compile(rf"{_N}\s*(?:[x×]\s*10\^?\s*|e\+?)(\d+)\s*FLOPs?\b", re.I)
BENCH_RE = re.compile(rf"\b{_N}\s?%\s+(?:accuracy\s+|pass@1\s+)?on\s+([A-Z][\w\-.]*[\w])")
# 저장소 이름의 크기 표기: Qwen3-30B-A3B -> total 30B, active 3B
NAME_SIZE_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)([BM])(?:-A(\d+(?:\.\d+)?)([BM]))?(?![a-z0-9])", re.I)
# 모델 카드 표/목록 표기: "| Total Parameters | 30.5B |", "- Number of Parameters: 7.6B"
CARD_PARAMS_RE = re.compile(
    rf"\b(total|activated|active)?\s*(?:number\s+of\s+)?param(?:eter)?s?\b(?:\s*\([^)]*\))?"
    rf"[\s*|:：]*{_N}\s?(B|M|T)\b(?!\s*tokens)",
    re.I,
)
# "2.5 billion active per token", "3.3B activated", "| Active parameters per token | Approximately 2.5B"
CARD_ACTIVE_RE = re.compile(
    rf"\b{_N}\s?(B|M|billion|million)\s+(?:active|activated)\b"
    rf"|activ(?:e|ated)\s+param(?:eter)?s?[^|\n:]{{0,20}}[*\s]*[|:][*\s]*(?:~|approximately|about|approx\.)?\s*{_N}\s?(B|M)\b",
    re.I,
)
# 최대 컨텍스트 선언만: "Context length: 131,072 tokens", "200K-token context window".
# "2T tokens at 32K context"(훈련 단계 설정)는 잡지 않는다.
CARD_CONTEXT_RE = re.compile(
    r"context\s+(?:length|window)[*\s|:]*(?:of\s+)?(?:up\s+to\s+)?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s?[KM])\b"
    r"|\b(\d+(?:\.\d+)?\s?[KM])(?:-token)?\s+(?:token\s+)?context\s+window",
    re.I,
)
MATH_HINT_RE = re.compile(
    r"\b(theorem|we prove|provabl[ey]|convergence rate|lower bound|upper bound|closed[- ]form)\b", re.I
)

_UNIT = {"billion": "B", "million": "M", "trillion": "T"}


def _qty(num: str, unit: str) -> str:
    return f"{num}{_UNIT.get(unit.lower(), unit.upper())}"


def _quote(text: str, m: re.Match) -> str:
    return text[max(0, m.start() - 40) : m.end() + 20].strip()


def repo_has_size(repo: str, size: str) -> bool:
    target = parse_quantity(size)
    name = repo.split("/")[-1]
    return any(parse_quantity(f"{n}{u}") == target for n, u, *_ in NAME_SIZE_RE.findall(name))


def name_sizes(repo: str) -> tuple[str | None, str | None]:
    m = NAME_SIZE_RE.search(repo.split("/")[-1])
    if not m:
        return None, None
    total = f"{m.group(1)}{m.group(2).upper()}"
    active = f"{m.group(3)}{m.group(4).upper()}" if m.group(3) else None
    return total, active


def clean_card(card: str) -> str:
    """YAML 머리말과 코드 블록(설치·예제 코드의 숫자)을 뺀 모델 카드 본문."""
    card = re.sub(r"\A---\n.*?\n---\n", "", card, flags=re.S)
    return re.sub(r"```.*?```", " ", card, flags=re.S)


def extract_model_item(item: Item, card: str | None = None) -> list[Claim]:
    """모델 출시 소식: 저장소 이름의 크기 표기와, 모델 카드에 저자가 선언한 수치가 주장이다."""
    repo = item.artifacts["models"][0]
    total, active = name_sizes(repo)
    claims = []
    if total:
        claims.append(
            Claim(
                "hf_model_params",
                f"{repo}의 총 파라미터 수는 약 {total} (이름 표기)",
                {"repo": repo, "value": total, "scope": "total"},
                quote=repo,
            )
        )
    if active:
        claims.append(
            Claim(
                "hf_model_params",
                f"{repo}의 활성 파라미터 수는 약 {active} (이름 표기)",
                {"repo": repo, "value": active, "scope": "active"},
                quote=repo,
            )
        )
    if card:
        claims += _card_claims(repo, clean_card(card))
    return _dedupe(claims)


def _card_claims(repo: str, text: str) -> list[Claim]:
    out = []
    seen_scope = set()
    for m in CARD_PARAMS_RE.finditer(text):
        qualifier = (m.group(1) or "total").lower()
        scope = "active" if qualifier.startswith("activ") else "total"
        if scope in seen_scope:
            continue
        # "(2B active)"처럼 바로 뒤에 active가 붙으면 활성 파라미터 표기
        if re.match(r"\s*(?:\(|,)?\s*(?:active|activated)", text[m.end() : m.end() + 12], re.I):
            scope = "active"
        seen_scope.add(scope)
        size = _qty(m.group(2), m.group(3))
        label = "활성" if scope == "active" else "총"
        out.append(
            Claim(
                "hf_model_params",
                f"모델 카드 선언: {label} 파라미터 {size}",
                {"repo": repo, "value": size, "scope": scope},
                quote=_quote(text, m),
            )
        )
    if "active" not in seen_scope and (m := CARD_ACTIVE_RE.search(text)):
        num, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        size = _qty(num, unit)
        out.append(
            Claim(
                "hf_model_params",
                f"모델 카드 선언: 활성 파라미터 {size}",
                {"repo": repo, "value": size, "scope": "active"},
                quote=_quote(text, m),
            )
        )
    if m := CARD_CONTEXT_RE.search(text):
        ctx = (m.group(1) or m.group(2)).replace(" ", "")
        out.append(
            Claim(
                "hf_model_context",
                f"모델 카드 선언: 컨텍스트 길이 {ctx}",
                {"repo": repo, "value": ctx},
                quote=_quote(text, m),
            )
        )
    return out


def extract_text_item(item: Item) -> list[Claim]:
    text = f"{item.title}. {item.summary}"
    models = item.artifacts.get("models", [])
    claims: list[Claim] = []

    def bind(size: str | None) -> str | None:
        """주장된 크기와 이름이 맞는 연결 모델, 없으면 연결 모델이 하나뿐일 때 그것."""
        if size:
            match = next((r for r in models if repo_has_size(r, size)), None)
            if match:
                return match
        return models[0] if len(models) == 1 else None

    sizes = []
    for m in PARAMS_RE.finditer(text):
        size = _qty(m.group(1), m.group(2))
        sizes.append(size)
        repo = bind(size)
        if repo:
            claims.append(
                Claim(
                    "hf_model_params",
                    f"{repo}는 {size} 파라미터 모델",
                    {"repo": repo, "value": size, "scope": "total"},
                    quote=_quote(text, m),
                )
            )
        else:
            claims.append(
                Claim("reported_params", f"{size} 파라미터 모델", {"value": size}, quote=_quote(text, m))
            )
    for m in ACTIVE_RE.finditer(text):
        size = _qty(m.group(1), m.group(2))
        repo = bind(sizes[0] if sizes else None)
        kind, params = (
            ("hf_model_params", {"repo": repo, "value": size, "scope": "active"})
            if repo
            else ("reported_params", {"value": size, "scope": "active"})
        )
        claims.append(Claim(kind, f"활성 파라미터 {size}", params, quote=_quote(text, m)))

    primary_repo = bind(sizes[0] if sizes else None)
    for m in TOKENS_RE.finditer(text):
        tokens = _qty(m.group(1), m.group(2))
        if primary_repo:
            claims.append(
                Claim(
                    "hf_training_tokens",
                    f"{primary_repo}는 {tokens} 토큰으로 훈련",
                    {"repo": primary_repo, "value": tokens},
                    quote=_quote(text, m),
                )
            )
        else:
            claims.append(
                Claim(
                    "reported_training_tokens",
                    f"{tokens} 토큰으로 훈련",
                    {"value": tokens},
                    quote=_quote(text, m),
                )
            )
    for m in CONTEXT_RE.finditer(text):
        num, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        ctx = _qty(num, unit)
        if primary_repo:
            claims.append(
                Claim(
                    "hf_model_context",
                    f"{primary_repo}의 컨텍스트 길이 {ctx}",
                    {"repo": primary_repo, "value": ctx},
                    quote=_quote(text, m),
                )
            )
        else:
            claims.append(
                Claim("reported_context", f"컨텍스트 길이 {ctx}", {"value": ctx}, quote=_quote(text, m))
            )

    flops = FLOPS_RE.search(text)
    tok = TOKENS_RE.search(text)
    if flops and tok and sizes:
        claims.append(
            Claim(
                "training_flops",
                "논문이 밝힌 N, D, 연산량이 6ND 관계로 서로 맞는다",
                {
                    "flops": f"{flops.group(1)}e{flops.group(2)}",
                    "params": sizes[0],
                    "tokens": _qty(tok.group(1), tok.group(2)),
                },
                quote=_quote(text, flops),
            )
        )
    for m in list(BENCH_RE.finditer(text))[:2]:
        claims.append(
            Claim(
                "benchmark_score",
                f"{m.group(2)}에서 {m.group(1)}%",
                {"benchmark": m.group(2), "value": float(m.group(1))},
                quote=_quote(text, m),
            )
        )

    for ds in item.artifacts.get("datasets", [])[:3]:
        claims.append(Claim("hf_dataset", f"공개 데이터셋 {ds}가 실제로 존재하고 접근 가능", {"repo": ds}))

    if MATH_HINT_RE.search(text) and "math-candidate" not in item.tags:
        item.tags.append("math-candidate")
    return _dedupe(claims)


def _dedupe(claims: list[Claim]) -> list[Claim]:
    return list({c.id: c for c in claims}.values())


def extract(item: Item, card: str | None = None) -> list[Claim]:
    if item.section == "model" and item.artifacts.get("models"):
        return extract_model_item(item, card)
    return extract_text_item(item)
