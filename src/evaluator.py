"""
evaluator.py

Runs an eval suite across a matrix of (model, provider) pairs and decides
whether a provider is ready to carry traffic.

The part that matters is the last step. Plenty of tools will print latency
numbers. The question a launch actually needs answered is binary: do we route
production traffic to this provider or not. So every run ends in a GO or
NO-GO against explicit, configurable gates, and the reasons are stated.

Quality is checked with assertions rather than model grading. For provider
qualification you are not asking "is this a good model," you already know the
model. You are asking "did this provider serve the model correctly." Assertions
catch the failures that actually happen: truncation, empty completions, broken
JSON, a quantized endpoint dropping instruction-following.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any

from client import CallResult


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def _check(assertion: dict[str, Any], text: str) -> tuple[bool, str]:
    """Evaluate one assertion against completion text."""
    kind = assertion.get("type")

    if kind == "contains":
        needle = assertion["value"]
        if not isinstance(needle, str):
            return False, "contains value must be a string"
        ok = needle.lower() in text.lower()
        return ok, f"expected to contain {needle!r}"

    if kind == "not_contains":
        needle = assertion["value"]
        if not isinstance(needle, str):
            return False, "not_contains value must be a string"
        ok = needle.lower() not in text.lower()
        return ok, f"expected NOT to contain {needle!r}"

    if kind == "regex":
        pattern = assertion.get("value")
        if not isinstance(pattern, str):
            return False, "regex value must be a string"
        try:
            ok = re.search(pattern, text, re.IGNORECASE | re.DOTALL) is not None
        except re.error as e:
            return False, f"invalid regex: {e}"
        return ok, f"expected to match /{pattern}/"

    if kind == "is_json":
        stripped = text.strip()
        # Models often wrap JSON in fences even when told not to.
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?|```$", "", stripped, flags=re.MULTILINE).strip()
        try:
            json.loads(stripped)
            return True, "valid JSON"
        except (json.JSONDecodeError, TypeError) as e:
            return False, f"invalid JSON: {getattr(e, 'msg', str(e))}"

    if kind == "json_object":
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?|```$", "", stripped, flags=re.MULTILINE).strip()
        try:
            actual = json.loads(stripped)
        except (json.JSONDecodeError, TypeError) as e:
            return False, f"invalid JSON: {getattr(e, 'msg', str(e))}"
        expected = assertion.get("value")
        ok = isinstance(actual, dict) and actual == expected
        return ok, f"expected JSON object {expected!r}"

    if kind == "min_length":
        try:
            limit = int(assertion["value"])
        except (KeyError, TypeError, ValueError):
            return False, "min_length value must be an integer"
        ok = len(text.strip()) >= limit
        return ok, f"expected at least {limit} chars, got {len(text.strip())}"

    if kind == "max_length":
        try:
            limit = int(assertion["value"])
        except (KeyError, TypeError, ValueError):
            return False, "max_length value must be an integer"
        ok = len(text.strip()) <= limit
        return ok, f"expected at most {limit} chars, got {len(text.strip())}"

    return False, f"unknown assertion type {kind!r}"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    model: str
    provider: str
    ok: bool
    ttft_ms: float | None
    total_ms: float
    tokens_per_sec: float
    completion_tokens: int
    cost: float | None
    model_served: str
    quality_passed: bool
    failed_assertions: list[str] = field(default_factory=list)
    error: str = ""
    text_preview: str = ""
    provider_served: str = ""


@dataclass
class ProviderReport:
    provider: str
    model: str
    runs: int
    errors: int
    error_rate: float
    quality_pass_rate: float
    ttft_p50: float | None
    ttft_p95: float | None
    latency_p50: float
    latency_p95: float
    mean_tokens_per_sec: float
    cost_per_1k_completion: float | None
    total_cost: float | None
    verdict: str = "PENDING"
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

DEFAULT_GATES = {
    "max_error_rate": 0.05,        # more than 5% failed requests is not launchable
    "min_quality_pass_rate": 0.95, # correctness is close to non-negotiable
    "max_ttft_p95_ms": 3000.0,
    "max_latency_p95_ms": 30000.0,
    "min_tokens_per_sec": 5.0,
}


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize(
    results: list[CaseResult], gates: dict[str, float] | None = None
) -> list[ProviderReport]:
    """Aggregate per (model, provider) and apply launch gates."""
    gates = {**DEFAULT_GATES, **(gates or {})}
    grouped: dict[tuple[str, str], list[CaseResult]] = {}
    for r in results:
        grouped.setdefault((r.model, r.provider), []).append(r)

    reports: list[ProviderReport] = []

    for (model, provider), rows in sorted(grouped.items()):
        good = [r for r in rows if r.ok]
        errors = len(rows) - len(good)
        error_rate = errors / len(rows) if rows else 1.0

        quality_pass = sum(1 for r in good if r.quality_passed)
        quality_rate = quality_pass / len(good) if good else 0.0

        ttfts = [r.ttft_ms for r in good if r.ttft_ms is not None]
        latencies = [r.total_ms for r in good]
        tps = [r.tokens_per_sec for r in good if r.tokens_per_sec > 0]

        costs = [r.cost for r in good if r.cost is not None]
        total_cost = sum(costs) if costs else None
        completion_tokens = sum(r.completion_tokens for r in good)
        cost_per_1k = (
            (total_cost / completion_tokens * 1000)
            if total_cost is not None and completion_tokens
            else None
        )

        report = ProviderReport(
            provider=provider,
            model=model,
            runs=len(rows),
            errors=errors,
            error_rate=error_rate,
            quality_pass_rate=quality_rate,
            ttft_p50=_pct(ttfts, 0.50),
            ttft_p95=_pct(ttfts, 0.95),
            latency_p50=_pct(latencies, 0.50) or 0.0,
            latency_p95=_pct(latencies, 0.95) or 0.0,
            mean_tokens_per_sec=statistics.fmean(tps) if tps else 0.0,
            cost_per_1k_completion=cost_per_1k,
            total_cost=total_cost,
        )

        reasons: list[str] = []

        if error_rate > gates["max_error_rate"]:
            reasons.append(
                f"error rate {error_rate:.1%} exceeds {gates['max_error_rate']:.0%}"
            )
        if quality_rate < gates["min_quality_pass_rate"]:
            reasons.append(
                f"quality pass rate {quality_rate:.1%} below "
                f"{gates['min_quality_pass_rate']:.0%}"
            )
        if report.ttft_p95 is not None and report.ttft_p95 > gates["max_ttft_p95_ms"]:
            reasons.append(
                f"TTFT p95 {report.ttft_p95:.0f}ms exceeds "
                f"{gates['max_ttft_p95_ms']:.0f}ms"
            )
        if report.latency_p95 > gates["max_latency_p95_ms"]:
            reasons.append(
                f"latency p95 {report.latency_p95:.0f}ms exceeds "
                f"{gates['max_latency_p95_ms']:.0f}ms"
            )
        if report.mean_tokens_per_sec < gates["min_tokens_per_sec"]:
            reasons.append(
                f"throughput {report.mean_tokens_per_sec:.1f} tok/s below "
                f"{gates['min_tokens_per_sec']:.0f} tok/s"
            )

        if not good:
            reasons.append("no successful responses")

        report.verdict = "NO-GO" if reasons else "GO"
        report.reasons = reasons
        reports.append(report)

    return reports


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_case(client, case: dict[str, Any], model: str, provider: str,
             fetch_cost: bool = True) -> CaseResult:
    """Execute one case against one (model, provider) pair."""
    messages = case.get("messages") or [
        {"role": "user", "content": case["prompt"]}
    ]

    call: CallResult = client.chat(
        model=model,
        messages=messages,
        provider=provider or None,
        max_tokens=case.get("max_tokens", 512),
        temperature=case.get("temperature", 0.0),
    )

    failed: list[str] = []
    quality_ok = True

    if call.ok:
        for assertion in case.get("assertions", []) or []:
            try:
                passed, description = _check(assertion, call.text)
            except (KeyError, TypeError, ValueError) as e:
                passed, description = False, f"invalid assertion: {e}"
            if not passed:
                quality_ok = False
                failed.append(description)
    else:
        quality_ok = False

    cost = None
    if fetch_cost and call.ok and call.generation_id:
        metadata = getattr(client, "fetch_generation", None)
        if metadata:
            generation = metadata(call.generation_id)
            if generation:
                cost_value = generation.get("total_cost")
                cost = float(cost_value) if cost_value is not None else None
                call.provider_served = (
                    generation.get("provider_name")
                    or generation.get("provider")
                    or call.provider_served
                )
        else:
            cost = client.fetch_cost(call.generation_id)

    return CaseResult(
        case_id=case["id"],
        model=model,
        provider=provider or "auto",
        ok=call.ok,
        ttft_ms=call.ttft_ms,
        total_ms=call.total_ms,
        tokens_per_sec=call.tokens_per_sec,
        completion_tokens=call.completion_tokens,
        cost=cost,
        model_served=call.model_served,
        quality_passed=quality_ok,
        failed_assertions=failed,
        error=call.error,
        text_preview=call.text[:160].replace("\n", " "),
        provider_served=call.provider_served,
    )


def run_matrix(
    client,
    suite: dict[str, Any],
    model: str,
    providers: list[str],
    repeats: int = 1,
    fetch_cost: bool = True,
    on_progress=None,
) -> list[CaseResult]:
    """Run every case against every provider, `repeats` times each."""
    results: list[CaseResult] = []
    cases = suite["cases"]
    total = len(providers) * len(cases) * repeats
    done = 0

    for provider in providers:
        for _ in range(repeats):
            for case in cases:
                res = run_case(client, case, model, provider, fetch_cost=fetch_cost)
                results.append(res)
                done += 1
                if on_progress:
                    on_progress(done, total, res)

    return results


def results_to_dicts(results: list[CaseResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]


def dicts_to_results(rows: list[dict[str, Any]]) -> list[CaseResult]:
    return [CaseResult(**row) for row in rows]
