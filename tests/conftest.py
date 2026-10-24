import json
import struct
from pathlib import Path

import pytest

from llm_news import http

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parent.parent


def fixture_json(name: str):
    return json.loads((FIXTURES / name).read_text())


class FakeWeb:
    """URL -> 응답 매핑. Range 헤더를 지원해 safetensors 헤더 읽기도 흉내 낸다."""

    def __init__(self):
        self.routes: dict[str, tuple[int, bytes]] = {}
        self.requests: list[tuple[str, dict]] = []

    def add(self, url: str, body, status: int = 200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        if isinstance(body, str):
            body = body.encode()
        self.routes[url] = (status, body)

    def __call__(self, url: str, headers: dict) -> tuple[int, bytes]:
        self.requests.append((url, headers))
        status, body = self.routes.get(url, (404, b"not found"))
        rng = headers.get("Range")
        if rng and status == 200:
            start, end = (int(x) for x in rng[6:].split("-"))
            return 206, body[start : end + 1]
        return status, body


@pytest.fixture
def web():
    fake = FakeWeb()
    http.set_transport(fake)
    yield fake
    http.set_transport(None)


def safetensors_blob(shapes: dict[str, list[int]]) -> bytes:
    """가중치 본문 없이 헤더만 가진 safetensors 바이트열 (테스트는 헤더만 읽는다)."""
    header = {
        name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]} for name, shape in shapes.items()
    }
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + b"\0" * 64


HUB = "https://huggingface.co"


def add_model(
    web: FakeWeb,
    repo: str,
    *,
    sha="abc1234def",
    total=None,
    config=None,
    card=None,
    siblings=None,
    extra=None,
):
    info = {"id": repo, "sha": sha, "siblings": [{"rfilename": s} for s in (siblings or [])]}
    if total is not None:
        info["safetensors"] = {"parameters": {"BF16": total}, "total": total}
    info.update(extra or {})
    web.add(f"{HUB}/api/models/{repo}", info)
    if config is not None:
        web.add(f"{HUB}/{repo}/resolve/{sha}/config.json", config)
    if card is not None:
        web.add(f"{HUB}/{repo}/resolve/{sha}/README.md", card)
    return sha
