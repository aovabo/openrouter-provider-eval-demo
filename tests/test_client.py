import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import client  # noqa: E402


class _Response:
    status = 200

    def __init__(self, lines, headers=None):
        self.lines = lines
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.lines)


def _run(lines, headers=None):
    with patch.object(
        client.urllib.request,
        "urlopen",
        lambda request, timeout: _Response(lines, headers),
    ):
        return client.OpenRouterClient("key").chat(
            "m", [{"role": "user", "content": "x"}], "p"
        )


def test_stream_requires_done_and_reads_header_generation_id():
    result = _run(
        [
            b'data: {"id":"body-id","model":"m","provider":"p","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n',
            b'\n',
            b'data: {"usage":{"completion_tokens":1}}\n',
            b'\n',
            b'data: [DONE]\n\n',
        ],
        {"X-Generation-Id": "header-id"},
    )
    assert result.ok is True
    assert result.generation_id == "body-id"
    assert result.provider_served == "p"


def test_stream_without_done_is_an_error():
    result = _run(
        [b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'],
    )
    assert result.ok is False
    assert "[DONE]" in result.error


def test_stream_error_is_an_error():
    result = _run(
        [
            b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
            b'data: {"error":{"code":503,"message":"provider unavailable"}}\n\n',
            b'data: [DONE]\n\n',
        ],
    )
    assert result.ok is False
    assert "503" in result.error


def _run_all():
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}  {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    return passed == len(fns)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)