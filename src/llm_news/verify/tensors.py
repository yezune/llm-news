"""safetensors 헤더만 읽어 실제 가중치의 텐서 이름과 shape를 얻는다 (가중치 본문은 받지 않음).

safetensors 파일 형식: [8바이트 little-endian 헤더 길이 N][N바이트 JSON 헤더][텐서 데이터].
샤드마다 앞의 8+N 바이트만 Range 요청으로 받으므로, 수백 GB 모델도 수 MB로 구조를 확인할 수 있다.
이 정보로 아키텍처 계산식이 없는 모델도 MoE 활성 파라미터를 원본 가중치 기준으로 센다.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .. import http

HUB = "https://huggingface.co"
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_SHARDS = 256
# 대부분의 샤드 헤더는 수십~수백 KB라 첫 요청 한 번으로 끝난다.
FIRST_CHUNK = 512 * 1024
WORKERS = 8
# 토큰당 활성 전문가 수를 나타내는 config 키 (모델마다 이름이 다르다)
TOP_K_KEYS = (
    "num_experts_per_tok",
    "num_experts_per_token",
    "moe_topk",
    "moe_top_k",
    "n_activated_experts",
    "top_k",
    "num_activated_experts",
)
# experts.<i>.  형태(전문가별 텐서) — shared_experts 는 앞이 '_'라 걸리지 않는다
EXPERT_RE = re.compile(r"^(.*\.experts)\.(\d+)\.")
# experts.gate_up_proj 처럼 전문가 축이 첫 차원으로 합쳐진 텐서
EMBED_RE = re.compile(r"(?:^|\.)(?:embed_tokens|wte|word_embeddings|tok_embeddings)\.weight$")
FUSED_RE = re.compile(r"^(.*\.experts)\.[^.\d][^.]*(?:\.weight)?$")


@dataclass
class TensorStats:
    total: int
    expert_total: int
    # 입력 임베딩 행렬 (집계 기준이 '임베딩 제외'인 주장과 비교할 때 쓴다)
    embedding: int
    active: int | None
    n_experts: int | None
    top_k: int | None
    shards: int


def shard_files(repo: str, revision: str, siblings: list[str]) -> list[str]:
    index = http.try_get_json(f"{HUB}/{repo}/resolve/{revision}/model.safetensors.index.json")
    if index and "weight_map" in index:
        return sorted(set(index["weight_map"].values()))
    if "model.safetensors" in siblings:
        return ["model.safetensors"]
    # 루트의 safetensors만 (consolidated.* 는 같은 가중치의 다른 포맷 사본)
    return sorted(
        s
        for s in siblings
        if s.endswith(".safetensors") and "/" not in s and not s.startswith("consolidated")
    )


def read_header(repo: str, revision: str, filename: str) -> dict[str, list[int]]:
    url = f"{HUB}/{repo}/resolve/{revision}/{filename}"
    head = http.get_range(url, 0, FIRST_CHUNK - 1)
    n = int.from_bytes(head[:8], "little")
    if not 0 < n <= MAX_HEADER_BYTES:
        raise ValueError(f"비정상 safetensors 헤더 길이 {n}: {filename}")
    raw = head[8 : 8 + n] if 8 + n <= len(head) else http.get_range(url, 8, 8 + n - 1)
    header = json.loads(raw)
    header.pop("__metadata__", None)
    return {name: meta["shape"] for name, meta in header.items()}


def top_k_from_config(config: dict[str, Any]) -> int | None:
    for cfg in (config.get("text_config") or {}, config):
        for key in TOP_K_KEYS:
            val = cfg.get(key)
            if isinstance(val, list) and val:
                val = val[0]
            if isinstance(val, int) and val > 0:
                return val
    return None


def tensor_stats(shapes: dict[str, list[int]], top_k: int | None, shards: int) -> TensorStats:
    total = sum(math.prod(s) for s in shapes.values())
    embedding = sum(math.prod(s) for name, s in shapes.items() if EMBED_RE.search(name))
    # 층(prefix)별 전문가 파라미터와 전문가 수
    per_prefix: dict[str, dict[str, int]] = defaultdict(lambda: {"params": 0, "experts": 0})
    indexed: dict[str, set[int]] = defaultdict(set)
    for name, shape in shapes.items():
        n = math.prod(shape)
        if m := EXPERT_RE.match(name):
            per_prefix[m.group(1)]["params"] += n
            indexed[m.group(1)].add(int(m.group(2)))
        elif (m := FUSED_RE.match(name)) and len(shape) == 3:
            per_prefix[m.group(1)]["params"] += n
            per_prefix[m.group(1)]["experts"] = max(per_prefix[m.group(1)]["experts"], shape[0])
    for prefix, idx in indexed.items():
        per_prefix[prefix]["experts"] = len(idx)
    expert_total = sum(v["params"] for v in per_prefix.values())
    n_experts = max((v["experts"] for v in per_prefix.values()), default=None)
    active = None
    if expert_total and top_k:
        routed = sum(v["params"] * min(top_k, v["experts"]) / v["experts"] for v in per_prefix.values())
        active = int(round(total - expert_total + routed))
    elif not expert_total:
        active = total  # 밀집 모델
    return TensorStats(total, expert_total, embedding, active, n_experts, top_k, shards)


def stats_from_hub(
    repo: str, revision: str, siblings: list[str], config: dict[str, Any] | None
) -> TensorStats | None:
    files = shard_files(repo, revision, siblings)
    if not files or len(files) > MAX_SHARDS:
        return None
    shapes: dict[str, list[int]] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for part in pool.map(lambda f: read_header(repo, revision, f), files):
            shapes.update(part)
    return tensor_stats(shapes, top_k_from_config(config or {}), len(files))
