"""
report.py

Turns eval results into something a human can act on.

Two outputs: a terminal table for the person running it, and a Markdown
report that can be pasted straight into a launch ticket or sent to the
provider. The Markdown version exists because provider qualification is a
conversation with the provider, and "your p95 TTFT is 4.2s against a 3s gate"
lands better than a screenshot.
"""

from __future__ import annotations

from datetime import datetime, timezone

from evaluator import CaseResult, ProviderReport

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _fmt_ms(v: float | None) -> str:
    return f"{v:,.0f}ms" if v is not None else "n/a"


def _fmt_cost(v: float | None) -> str:
    return f"${v:.5f}" if v is not None else "n/a"


def terminal_report(reports: list[ProviderReport], color: bool = True) -> str:
    def c(code: str) -> str:
        return code if color else ""

    lines: list[str] = []
    lines.append("")
    lines.append(f"{c(BOLD)}PROVIDER COMPARISON{c(RESET)}")
    lines.append("=" * 96)

    header = (
        f"{'PROVIDER':<22}{'VERDICT':<10}{'ERR':>6}{'QUAL':>7}"
        f"{'TTFT p50':>11}{'TTFT p95':>11}{'LAT p95':>11}{'TOK/S':>8}{'$/1K out':>10}"
    )
    lines.append(header)
    lines.append("-" * 96)

    for r in sorted(reports, key=lambda x: (x.verdict != "GO", x.ttft_p95 or 9e9)):
        badge = f"{c(GREEN)}GO{c(RESET)}" if r.verdict == "GO" else f"{c(RED)}NO-GO{c(RESET)}"
        pad = 10 - len("GO" if r.verdict == "GO" else "NO-GO")
        cost = (
            f"${r.cost_per_1k_completion:.4f}"
            if r.cost_per_1k_completion is not None
            else "n/a"
        )
        lines.append(
            f"{r.provider:<22}{badge}{' ' * pad}"
            f"{r.error_rate:>5.0%} "
            f"{r.quality_pass_rate:>6.0%} "
            f"{_fmt_ms(r.ttft_p50):>10} "
            f"{_fmt_ms(r.ttft_p95):>10} "
            f"{_fmt_ms(r.latency_p95):>10} "
            f"{r.mean_tokens_per_sec:>7.1f} "
            f"{cost:>9}"
        )

    lines.append("-" * 96)

    blockers = [r for r in reports if r.verdict == "NO-GO"]
    if blockers:
        lines.append("")
        lines.append(f"{c(BOLD)}BLOCKERS{c(RESET)}")
        for r in blockers:
            lines.append(f"  {c(RED)}{r.provider}{c(RESET)}")
            for reason in r.reasons:
                lines.append(f"    - {reason}")

    passing = [r for r in reports if r.verdict == "GO"]
    if passing:
        best = min(passing, key=lambda x: x.ttft_p95 or 9e9)
        cheapest = [p for p in passing if p.cost_per_1k_completion is not None]
        lines.append("")
        lines.append(f"{c(BOLD)}RECOMMENDATION{c(RESET)}")
        lines.append(
            f"  Fastest passing provider: {c(GREEN)}{best.provider}{c(RESET)} "
            f"(TTFT p95 {_fmt_ms(best.ttft_p95)})"
        )
        if cheapest:
            cheap = min(cheapest, key=lambda x: x.cost_per_1k_completion)
            lines.append(
                f"  Cheapest passing provider: {c(GREEN)}{cheap.provider}{c(RESET)} "
                f"(${cheap.cost_per_1k_completion:.4f} per 1K completion tokens)"
            )
        if len(passing) > 1:
            lines.append(
                f"  {c(DIM)}Suggested routing order: "
                f"{', '.join(p.provider for p in sorted(passing, key=lambda x: x.ttft_p95 or 9e9))}{c(RESET)}"
            )

    lines.append("")
    return "\n".join(lines)


def markdown_report(
    reports: list[ProviderReport], results: list[CaseResult], model: str
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    passing = [r for r in reports if r.verdict == "GO"]

    out: list[str] = []
    out.append(f"# Provider qualification: `{model}`")
    out.append("")
    out.append(f"**Run:** {ts}  ")
    out.append(f"**Providers evaluated:** {len(reports)}  ")
    out.append(f"**Passing launch gates:** {len(passing)} of {len(reports)}")
    out.append("")

    out.append("## Summary")
    out.append("")
    out.append(
        "| Provider | Verdict | Errors | Quality | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | $ / 1K out |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(reports, key=lambda x: (x.verdict != "GO", x.ttft_p95 or 9e9)):
        cost = (
            f"${r.cost_per_1k_completion:.4f}"
            if r.cost_per_1k_completion is not None
            else "n/a"
        )
        verdict = "**GO**" if r.verdict == "GO" else "**NO-GO**"
        out.append(
            f"| `{r.provider}` | {verdict} | {r.error_rate:.0%} | {r.quality_pass_rate:.0%} "
            f"| {_fmt_ms(r.ttft_p50)} | {_fmt_ms(r.ttft_p95)} | {_fmt_ms(r.latency_p95)} "
            f"| {r.mean_tokens_per_sec:.1f} | {cost} |"
        )
    out.append("")

    blockers = [r for r in reports if r.verdict == "NO-GO"]
    if blockers:
        out.append("## Blockers")
        out.append("")
        for r in blockers:
            out.append(f"### `{r.provider}`")
            for reason in r.reasons:
                out.append(f"- {reason}")
            failures = [
                res
                for res in results
                if res.provider == r.provider and not res.quality_passed
            ]
            if failures:
                out.append("")
                out.append("Failed cases:")
                for f in failures[:5]:
                    detail = f.error or "; ".join(f.failed_assertions) or "unknown"
                    out.append(f"- `{f.case_id}`: {detail}")
            out.append("")

    if passing:
        out.append("## Recommendation")
        out.append("")
        best = min(passing, key=lambda x: x.ttft_p95 or 9e9)
        out.append(
            f"- Fastest passing provider: `{best.provider}` "
            f"(TTFT p95 {_fmt_ms(best.ttft_p95)})"
        )
        cheapest = [p for p in passing if p.cost_per_1k_completion is not None]
        if cheapest:
            cheap = min(cheapest, key=lambda x: x.cost_per_1k_completion)
            out.append(
                f"- Cheapest passing provider: `{cheap.provider}` "
                f"(${cheap.cost_per_1k_completion:.4f} per 1K completion tokens)"
            )
        order = ", ".join(
            f"`{p.provider}`"
            for p in sorted(passing, key=lambda x: x.ttft_p95 or 9e9)
        )
        out.append(f"- Suggested `provider.order`: {order}")
        out.append("")

    out.append("## Gates applied")
    out.append("")
    out.append("Error rate, quality pass rate, TTFT p95, latency p95, and throughput.")
    out.append("Gates are configurable per launch; defaults are conservative.")
    out.append("")

    return "\n".join(out)
