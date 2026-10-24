"""DeepSeek-V2 Multi-head Latent Attention의 가중치 흡수 (arXiv:2405.04434).

새 논문의 알고리즘 주장을 저장소 안에서 재현하는 방법의 예시다.
@algorithm("이름")으로 등록하면 claims/*.yaml 에서 kind: algorithm, params.name 으로 참조할 수 있다.
"""

import numpy as np

from llm_news.verify.algorithms import CheckResult, algorithm


@algorithm("mla_weight_absorption")
def mla_weight_absorption(rng, seq_len=16, d_model=64, d_latent=16, n_heads=4, head_dim=16):
    """MLA: 잠재 KV c만 캐시하고 W_UK를 쿼리에 흡수해도 어텐션 점수가 같다."""
    x = rng.standard_normal((seq_len, d_model))
    W_dkv = rng.standard_normal((d_model, d_latent)) / np.sqrt(d_model)  # 압축(다운) 투영
    W_uk = rng.standard_normal((n_heads, head_dim, d_latent)) / np.sqrt(d_latent)  # 키 복원(업) 투영
    W_q = rng.standard_normal((n_heads, d_model, head_dim)) / np.sqrt(d_model)

    c = x @ W_dkv  # 캐시되는 잠재 벡터: 토큰당 d_latent개
    q = np.einsum("td,hdk->htk", x, W_q)
    # 명시적 방식: 키를 복원한 뒤 점수 계산
    k = np.einsum("hkl,tl->htk", W_uk, c)
    explicit = np.einsum("hik,hjk->hij", q, k)
    # 흡수 방식: W_UK^T를 쿼리 쪽으로 옮기고 잠재 벡터와 바로 내적
    q_absorbed = np.einsum("htk,hkl->htl", q, W_uk)
    absorbed = np.einsum("hil,jl->hij", q_absorbed, c)

    err = float(np.abs(explicit - absorbed).max())
    cache_ratio = d_latent / (2 * n_heads * head_dim)
    return CheckResult(err < 1e-10, err, {"kv_cache_ratio_vs_mha": cache_ratio})
