"""
client.py

A minimal OpenRouter client built on the standard library only.

Two design choices worth calling out:

    1. No dependencies. urllib instead of requests, so the harness runs on
    Python 3.10+ with nothing to pip install first. When you are handing
     an eval tool to a provider partner to run on their side, "just run it"
     matters more than elegant HTTP code.

  2. Streaming by default, to measure time to first token. Total latency alone
     hides the difference between a provider that starts responding in 200ms
     and one that stalls for two seconds and then dumps the whole completion.
     For a user-facing product that difference is the whole experience, so
     TTFT is the metric that actually separates providers.

Cost comes from GET /api/v1/generation?id=..., which is the ground truth after
the fact. Estimating cost from token counts and a pricing table drifts; the
generation endpoint is what you were actually billed.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

BASE_URL = "https://openrouter.ai/api/v1"


def _sse_payloads(response):
    data_lines: list[str] = []
    buffer = ""

    def process_line(line: str):
        if not line:
            if data_lines:
                payload = "\n".join(data_lines).strip()
                data_lines.clear()
                return payload
            return None
        if line.startswith(":") or not line.startswith("data:"):
            return None
        data_lines.append(line[5:].lstrip(" "))
        return None

    for raw in response:
        buffer += raw.decode("utf-8", errors="replace")
        lines = buffer.split("\n")
        buffer = lines.pop()
        for line in lines:
            payload = process_line(line.rstrip("\r"))
            if payload is not None:
                yield payload
    if buffer:
        payload = process_line(buffer.rstrip("\r"))
        if payload is not None:
            yield payload
    if data_lines:
        yield "\n".join(data_lines).strip()


def _format_stream_error(error: Any) -> str:
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message") or "unknown provider error"
        return f"Stream error{f' {code}' if code is not None else ''}: {message}"
    return f"Stream error: {error}"


@dataclass
class CallResult:
    """Everything one request tells us."""

    ok: bool
    model_requested: str
    provider_requested: str | None

    # What actually served the request. Worth comparing against what we asked
    # for, since a fallback can silently change who answered.
    model_served: str = ""
    provider_served: str = ""
    generation_id: str = ""

    text: str = ""
    ttft_ms: float | None = None
    total_ms: float = 0.0

    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_sec: float = 0.0

    total_cost: float | None = None
    finish_reason: str = ""

    error: str = ""
    http_status: int | None = None


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        referer: str = "https://github.com/aovabo/openrouter-provider-eval",
        title: str = "openrouter-provider-eval",
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # OpenRouter uses these for attribution on the rankings page.
            "HTTP-Referer": referer,
            "X-Title": title,
        }

    # -- public ------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        provider: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        allow_fallbacks: bool = False,
    ) -> CallResult:
        """
        Run one streamed chat completion, optionally pinned to a provider.

        Pinning uses provider.only plus allow_fallbacks=false. Without
        disabling fallbacks the router is free to serve the request from
        somewhere else, which quietly invalidates a provider comparison.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        if provider:
            body["provider"] = {
                "only": [provider],
                "allow_fallbacks": allow_fallbacks,
            }

        result = CallResult(
            ok=False, model_requested=model, provider_requested=provider
        )

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers,
            method="POST",
        )

        start = time.perf_counter()
        chunks: list[str] = []
        saw_done = False

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result.http_status = resp.status
                result.generation_id = resp.headers.get("X-Generation-Id", "")
                for payload in _sse_payloads(resp):
                    if payload == "[DONE]":
                        saw_done = True
                        break

                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        result.error = "Malformed SSE event"
                        continue

                    result.generation_id = event.get("id") or result.generation_id
                    result.model_served = event.get("model") or result.model_served
                    result.provider_served = (
                        event.get("provider")
                        or event.get("provider_name")
                        or result.provider_served
                    )

                    event_error = event.get("error")
                    if event_error:
                        result.error = _format_stream_error(event_error)

                    for choice in event.get("choices", []) or []:
                        choice_error = choice.get("error")
                        if choice_error:
                            result.error = _format_stream_error(choice_error)
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            if result.ttft_ms is None:
                                result.ttft_ms = (time.perf_counter() - start) * 1000
                            chunks.append(delta)
                        if choice.get("finish_reason"):
                            result.finish_reason = choice["finish_reason"]

                    # Usage arrives once, in the final chunk before [DONE].
                    usage = event.get("usage")
                    if usage:
                        result.prompt_tokens = usage.get("prompt_tokens", 0) or 0
                        result.completion_tokens = usage.get("completion_tokens", 0) or 0

            result.total_ms = (time.perf_counter() - start) * 1000
            result.text = "".join(chunks)
            if not saw_done and not result.error:
                result.error = "Stream ended before [DONE]"
            if result.finish_reason == "error" and not result.error:
                result.error = "Provider returned finish_reason=error"
            result.ok = saw_done and not result.error and result.finish_reason != "error"

            if result.completion_tokens and result.total_ms > 0:
                result.tokens_per_sec = result.completion_tokens / (
                    result.total_ms / 1000
                )

        except urllib.error.HTTPError as e:
            result.total_ms = (time.perf_counter() - start) * 1000
            result.http_status = e.code
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # noqa: BLE001
                pass
            result.error = f"HTTP {e.code}: {detail or e.reason}"
        except urllib.error.URLError as e:
            result.total_ms = (time.perf_counter() - start) * 1000
            result.error = f"Connection error: {e.reason}"
        except TimeoutError:
            result.total_ms = (time.perf_counter() - start) * 1000
            result.error = f"Timed out after {self.timeout}s"
        except Exception as e:  # noqa: BLE001 -- an eval harness must never die mid-run
            result.total_ms = (time.perf_counter() - start) * 1000
            result.error = f"{type(e).__name__}: {e}"

        return result

    def fetch_generation(
        self, generation_id: str, retries: int = 4, delay: float = 0.8
    ) -> dict[str, Any] | None:
        """Pull generation metadata, retrying while asynchronous stats settle."""
        if not generation_id:
            return None

        url = f"{self.base_url}/generation?{urllib.parse.urlencode({'id': generation_id})}"

        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers=self._headers, method="GET")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                payload = data.get("data", data)
                if isinstance(payload, dict) and payload.get("total_cost") is not None:
                    return payload
            except Exception:  # noqa: BLE001
                pass
            time.sleep(delay * (attempt + 1))

        return None

    def fetch_cost(
        self, generation_id: str, retries: int = 4, delay: float = 0.8
    ) -> float | None:
        """
        Pull true billed cost from the generation endpoint.

        Stats are not always available the instant a stream closes, so this
        retries with a short backoff rather than reporting a null cost that
        would skew the comparison.
        """
        payload = self.fetch_generation(generation_id, retries=retries, delay=delay)
        return float(payload["total_cost"]) if payload else None

    def list_endpoints(self, model: str) -> list[str]:
        """
        List provider slugs currently serving a model.

        Useful when you are onboarding and do not yet know the slug, since the
        slug is what provider.only expects and guessing it silently fails.
        """
        author, _, slug = model.partition("/")
        if not slug:
            return []
        url = f"{self.base_url}/models/{author}/{slug}/endpoints"
        try:
            req = urllib.request.Request(url, headers=self._headers, method="GET")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return []

        payload = data.get("data", data) or {}
        endpoints = payload.get("endpoints", []) or []
        slugs = []
        for ep in endpoints:
            tag = ep.get("tag") or ep.get("provider_name") or ep.get("name")
            if tag:
                slugs.append(str(tag))
        return slugs
