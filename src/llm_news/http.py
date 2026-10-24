"""외부 출처 접근을 한 곳으로 모은 HTTP 계층.

테스트는 set_transport()로 네트워크 없이 픽스처를 주입한다.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import requests

USER_AGENT = "llm-news-curator/0.1 (daily LLM news verification)"
TIMEOUT = 30
RETRY_STATUS = (429, 500, 502, 503, 504)
# 429는 출처가 속도를 늦추라는 신호다. arXiv는 요청 간 3초 이상을 요구한다.
BACKOFF_SECONDS = (3, 10, 30)

# (url, headers) -> (status_code, body_bytes)
Transport = Callable[[str, dict[str, str]], tuple[int, bytes]]


class FetchError(RuntimeError):
    def __init__(self, url: str, status: int):
        super().__init__(f"HTTP {status}: {url}")
        self.url = url
        self.status = status


def _default_transport(url: str, extra: dict[str, str]) -> tuple[int, bytes]:
    headers = {"User-Agent": USER_AGENT, **extra}
    token = os.environ.get("HF_TOKEN")
    if token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {token}"
    last_exc: Exception | None = None
    resp = None
    # Range 요청은 서버가 Range를 무시하고 전체(수백 GB 가중치)를 보내도 요청한 길이까지만 읽는다.
    limit = _range_length(extra.get("Range"))
    for wait in (*BACKOFF_SECONDS, None):
        try:
            resp = requests.get(url, headers=headers, timeout=TIMEOUT, stream=limit is not None)
            if resp.status_code not in RETRY_STATUS:
                if limit is None:
                    return resp.status_code, resp.content
                body = resp.raw.read(limit, decode_content=True)
                resp.close()
                return resp.status_code, body
        except requests.RequestException as exc:
            last_exc = exc
        if wait is None:
            break
        retry_after = resp.headers.get("Retry-After", "") if resp is not None else ""
        time.sleep(int(retry_after) if retry_after.isdigit() else wait)
    if resp is None:
        assert last_exc is not None
        raise last_exc
    return resp.status_code, resp.content


def _range_length(spec: str | None) -> int | None:
    """읽을 최대 바이트 수. 서버가 Range를 무시하고 0부터 보내는 경우에도 end까지 담기도록 end+1."""
    if not spec or not spec.startswith("bytes="):
        return None
    _, _, end = spec[6:].partition("-")
    return int(end) + 1


_transport: Transport = _default_transport
_cache: dict[tuple[str, tuple[tuple[str, str], ...]], tuple[int, bytes]] = {}


def set_transport(transport: Transport | None) -> None:
    global _transport
    _transport = transport or _default_transport
    _cache.clear()


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def get_bytes(url: str, headers: dict[str, str] | None = None) -> bytes:
    key = (url, tuple(sorted((headers or {}).items())))
    if key not in _cache:
        _cache[key] = _transport(url, headers or {})
    status, body = _cache[key]
    if status not in (200, 206):
        raise FetchError(url, status)
    return body


def get_range(url: str, start: int, end: int) -> bytes:
    """바이트 구간 [start, end]만 받는다. safetensors 헤더를 가중치 없이 읽는 데 쓴다."""
    body = get_bytes(url, {"Range": f"bytes={start}-{end}"})
    want = end - start + 1
    # 206이면 정확히 want 바이트, Range를 무시한 200이면 0부터 온 것이므로 start 이전을 잘라낸다.
    return body if len(body) <= want else body[start : start + want]


def get_text(url: str) -> str:
    return get_bytes(url).decode("utf-8", errors="replace")


def get_json(url: str) -> Any:
    return json.loads(get_bytes(url))


def try_get_json(url: str) -> Any | None:
    try:
        return get_json(url)
    except (FetchError, ValueError, requests.RequestException):
        return None
