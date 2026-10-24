"""잘못된 추측 디코딩 변형 — 검증기가 틀린 알고리즘 주장을 잡아내는지 보여 주는 반례."""

import numpy as np

from llm_news.verify.algorithms import CheckResult, algorithm


@algorithm("speculative_resample_from_target")
def speculative_resample_from_target(rng, vocab=50, draft_quality=0.5):
    """거절 시 잔차 대신 목표 분포 p에서 다시 뽑는 변형이 p를 보존하는가 (보존하지 않음)."""
    p = rng.dirichlet(np.ones(vocab))
    q = draft_quality * p + (1 - draft_quality) * rng.dirichlet(np.ones(vocab))
    accept = np.minimum(p, q)
    out = accept + (1 - accept.sum()) * p
    err = float(np.abs(out - p).max())
    return CheckResult(err < 1e-12, err, {"acceptance_rate": float(accept.sum())})
