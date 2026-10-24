"""config.json만으로 디코더 전용 트랜스포머의 파라미터 수를 독립적으로 계산한다.

safetensors 헤더에서 집계한 실측 파라미터 수와 교차 검증하는 데 쓴다.
두 값이 서로 독립적인 경로(아키텍처 명세 vs 실제 가중치)에서 나오므로,
일치하면 "명시된 아키텍처 = 실제 가중치"까지 확인된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# 게이트형 MLP(gate/up/down 3개 행렬)를 쓰는 계열
GATED_MLP_TYPES = {
    "llama",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "qwen3_moe",
    "gemma",
    "gemma2",
    "gemma3_text",
    "phi3",
    "olmo",
    "olmo2",
    "granite",
    "smollm3",
}
# q/k/v 투영에 bias가 있는 계열
QKV_BIAS_TYPES = {"qwen2", "qwen2_moe"}
# head_dim 크기의 q_norm/k_norm이 있는 계열
QK_NORM_TYPES = {"qwen3", "qwen3_moe", "gemma3_text"}
# q_norm/k_norm이 헤드 전체 투영 차원(nh·hd, nkv·hd)에 걸리는 계열
FULL_QK_NORM_TYPES = {"olmo2"}
# 층마다 RMSNorm이 4개(어텐션/FFN 전후)인 계열
FOUR_NORM_TYPES = {"gemma2", "gemma3_text"}
# tie_word_embeddings가 명시되지 않았을 때 기본값이 True인 계열
TIED_BY_DEFAULT = {"gemma", "gemma2", "gemma3_text"}


@dataclass
class ParamCount:
    total: int
    embedding: int
    # MoE에서 토큰당 실제로 쓰이는 파라미터 수 (dense면 total과 같음)
    active: int

    @property
    def non_embedding(self) -> int:
        return self.total - self.embedding


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    """멀티모달 모델은 언어 모델 설정이 text_config 아래에 있다."""
    if "text_config" in config and isinstance(config["text_config"], dict):
        merged = dict(config["text_config"])
        merged.setdefault("model_type", config.get("model_type"))
        return merged
    return config


def count_params(config: dict[str, Any]) -> ParamCount | None:
    """지원하지 않는 아키텍처면 None (추측하지 않는다)."""
    cfg = text_config(config)
    mtype = cfg.get("model_type")
    if mtype not in GATED_MLP_TYPES:
        # 예: DeepSeek MLA는 저랭크 투영 구조가 달라 별도 공식이 필요하다.
        return None
    try:
        V = int(cfg["vocab_size"])
        H = int(cfg["hidden_size"])
        L = int(cfg["num_hidden_layers"])
        nh = int(cfg["num_attention_heads"])
    except (KeyError, TypeError, ValueError):
        return None
    nkv = int(cfg.get("num_key_value_heads") or nh)
    hd = int(cfg.get("head_dim") or H // nh)
    tied = cfg.get("tie_word_embeddings", mtype in TIED_BY_DEFAULT)

    attn = H * nh * hd + 2 * H * nkv * hd + nh * hd * H
    if cfg.get("attention_bias") or mtype in QKV_BIAS_TYPES:
        attn += nh * hd + 2 * nkv * hd
        if cfg.get("attention_bias") and mtype not in QKV_BIAS_TYPES:
            attn += H  # o_proj bias
    if mtype in QK_NORM_TYPES:
        attn += 2 * hd
    elif mtype in FULL_QK_NORM_TYPES:
        attn += nh * hd + nkv * hd
    norms = (4 if mtype in FOUR_NORM_TYPES else 2) * H

    n_experts = int(
        cfg.get("num_local_experts") or cfg.get("num_experts") or cfg.get("n_routed_experts") or 0
    )
    dense_mlp = 3 * H * int(cfg.get("intermediate_size") or 0)
    if n_experts:
        expert_I = int(cfg.get("moe_intermediate_size") or cfg["intermediate_size"])
        expert = 3 * H * expert_I
        router = H * n_experts
        shared_I = int(cfg.get("shared_expert_intermediate_size") or 0)
        shared = 3 * H * shared_I + (H if shared_I else 0)  # 공유 전문가 게이트
        top_k = int(cfg.get("num_experts_per_tok") or 1)
        sparse_layers = _sparse_layers(cfg, L)
        moe_total = n_experts * expert + router + shared
        moe_active = top_k * expert + router + shared
        mlp_total = sparse_layers * moe_total + (L - sparse_layers) * dense_mlp
        mlp_active = sparse_layers * moe_active + (L - sparse_layers) * dense_mlp
    else:
        mlp_total = mlp_active = L * dense_mlp

    embedding = V * H
    head = 0 if tied else V * H
    base = embedding + head + L * (attn + norms) + H
    return ParamCount(total=base + mlp_total, embedding=embedding, active=base + mlp_active)


def _sparse_layers(cfg: dict[str, Any], L: int) -> int:
    only_dense = set(cfg.get("mlp_only_layers") or [])
    step = int(cfg.get("decoder_sparse_step") or 1)
    first_dense = int(cfg.get("first_k_dense_replace") or 0)
    return sum(1 for i in range(L) if i not in only_dense and i >= first_dense and (i + 1) % step == 0)


def context_lengths(config: dict[str, Any]) -> list[int]:
    """설정이 허용하는 컨텍스트 길이 후보들.

    RoPE 스케일링 표기가 모델마다 달라서(확장 후 길이를 적거나, 원래 길이와 factor를 적거나)
    가능한 해석을 모두 돌려주고, 주장이 그중 하나와 맞는지 본다.
    """
    cfg = text_config(config)
    base = cfg.get("max_position_embeddings")
    if base is None:
        return []
    out = {int(base)}
    scaling = cfg.get("rope_scaling")
    if isinstance(scaling, dict) and scaling.get("factor"):
        factor = float(scaling["factor"])
        orig = scaling.get("original_max_position_embeddings")
        out.add(int((orig or base) * factor))
    return sorted(out)
