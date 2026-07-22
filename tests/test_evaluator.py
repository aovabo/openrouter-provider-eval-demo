"""
Tests for the assertion engine and the launch gates.

The gate logic is the part worth testing hardest, because a wrong verdict is
worse than no verdict. A false GO routes production traffic to a provider that
should not have it.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluator import CaseResult, _check, _pct, summarize  # noqa: E402


def _row(provider, ok=True, quality=True, ttft=300.0, total=800.0,
         tps=50.0, cost=0.0001, tokens=20, case="c1"):
    return CaseResult(
        case_id=case, model="m", provider=provider, ok=ok, ttft_ms=ttft,
        total_ms=total, tokens_per_sec=tps, completion_tokens=tokens,
        cost=cost, model_served="m", quality_passed=quality,
    )


# -- assertions ------------------------------------------------------------

def test_contains_is_case_insensitive():
    ok, _ = _check({"type": "contains", "value": "READY"}, "ready")
    assert ok is True


def test_not_contains():
    ok, _ = _check({"type": "not_contains", "value": "I cannot"}, "Sure, here you go")
    assert ok is True
    ok, _ = _check({"type": "not_contains", "value": "I cannot"}, "I cannot help")
    assert ok is False


def test_is_json_accepts_fenced_json():
    # Models wrap JSON in fences constantly, even when told not to. A provider
    # should not be failed for that, so the check strips fences first.
    ok, _ = _check({"type": "is_json"}, '```json\n{"a": 1}\n```')
    assert ok is True


def test_is_json_rejects_malformed():
    ok, _ = _check({"type": "is_json"}, "{status: ok}")
    assert ok is False


def test_regex_spans_newlines():
    ok, _ = _check({"type": "regex", "value": r"1\s*,.*20"}, "1,\n2,\n...\n20")
    assert ok is True


def test_length_bounds():
    ok, _ = _check({"type": "max_length", "value": 5}, "abc")
    assert ok is True
    ok, _ = _check({"type": "max_length", "value": 2}, "abc")
    assert ok is False


# -- percentiles -----------------------------------------------------------

def test_percentile_single_value():
    assert _pct([42.0], 0.95) == 42.0


def test_percentile_interpolates():
    assert _pct([0.0, 10.0], 0.5) == 5.0


def test_percentile_empty_is_none():
    assert _pct([], 0.5) is None


# -- gates -----------------------------------------------------------------

def test_clean_provider_gets_go():
    rows = [_row("good") for _ in range(10)]
    report = summarize(rows)[0]
    assert report.verdict == "GO"
    assert report.reasons == []


def test_slow_ttft_blocks_launch():
    rows = [_row("slow", ttft=5000.0) for _ in range(10)]
    report = summarize(rows)[0]
    assert report.verdict == "NO-GO"
    assert any("TTFT" in r for r in report.reasons)


def test_quality_failures_block_launch():
    # Half the responses are wrong. Fast and cheap does not matter.
    rows = [_row("quant", quality=(i % 2 == 0)) for i in range(10)]
    report = summarize(rows)[0]
    assert report.verdict == "NO-GO"
    assert any("quality" in r for r in report.reasons)


def test_error_rate_blocks_launch():
    rows = [_row("flaky", ok=(i > 1)) for i in range(10)]
    report = summarize(rows)[0]
    assert report.verdict == "NO-GO"
    assert any("error rate" in r for r in report.reasons)


def test_total_failure_is_reported_not_crashed():
    rows = [_row("dead", ok=False) for _ in range(5)]
    report = summarize(rows)[0]
    assert report.verdict == "NO-GO"
    assert "no successful responses" in report.reasons


def test_missing_throughput_blocks_launch():
    report = summarize([_row("unknown", tps=0.0)])[0]
    assert report.verdict == "NO-GO"
    assert any("throughput" in reason for reason in report.reasons)


def test_exact_json_object_assertion():
    ok, _ = _check(
        {"type": "json_object", "value": {"status": "ok", "count": 3}},
        '{"status":"ok","count":3}',
    )
    assert ok is True
    ok, _ = _check(
        {"type": "json_object", "value": {"status": "ok", "count": 3}},
        '{"status":"ok","count":30}',
    )
    assert ok is False


def test_malformed_assertions_fail_without_raising():
    ok, message = _check({"type": "regex", "value": "["}, "text")
    assert ok is False
    assert "invalid regex" in message
    ok, message = _check({"type": "contains", "value": 3}, "text")
    assert ok is False
    assert "string" in message


def test_gates_are_overridable():
    rows = [_row("slow", ttft=5000.0) for _ in range(10)]
    strict = summarize(rows)[0]
    assert strict.verdict == "NO-GO"
    relaxed = summarize(rows, {"max_ttft_p95_ms": 9000.0})[0]
    assert relaxed.verdict == "GO"


def test_cost_per_1k_completion_tokens():
    # 20 tokens at $0.0001 each run, 5 runs = $0.0005 over 100 tokens
    rows = [_row("p", cost=0.0001, tokens=20) for _ in range(5)]
    report = summarize(rows)[0]
    assert abs(report.cost_per_1k_completion - 0.005) < 1e-9


def test_providers_are_reported_separately():
    rows = [_row("a") for _ in range(5)] + [_row("b", ttft=9000.0) for _ in range(5)]
    reports = summarize(rows)
    assert len(reports) == 2
    verdicts = {r.provider: r.verdict for r in reports}
    assert verdicts["a"] == "GO"
    assert verdicts["b"] == "NO-GO"


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
