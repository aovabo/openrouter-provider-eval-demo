"""
cli.py

Command line interface for the provider eval harness.

    # Offline demo, no API key needed, runs instantly on bundled sample data
    python cli.py --demo

    # Discover which providers currently serve a model
    python cli.py --model meta-llama/llama-3.3-70b-instruct --list-providers

    # Live run against real providers
    python cli.py --model meta-llama/llama-3.3-70b-instruct \\
        --providers together,deepinfra,fireworks --repeats 3

    # Write a Markdown report for the launch ticket
    python cli.py --demo --markdown report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from client import OpenRouterClient  # noqa: E402
from evaluator import (  # noqa: E402
    DEFAULT_GATES,
    dicts_to_results,
    results_to_dicts,
    run_matrix,
    summarize,
)
from report import markdown_report, terminal_report  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SUITE = os.path.join(HERE, "evals", "default_suite.json")
SAMPLE_RESULTS = os.path.join(HERE, "samples", "offline_results.json")


def _progress(done: int, total: int, res) -> None:
    mark = "." if res.ok and res.quality_passed else ("q" if res.ok else "x")
    sys.stdout.write(mark)
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare OpenRouter providers serving the same model and "
        "decide whether each is ready to carry traffic."
    )
    p.add_argument("--model", help="Model slug, e.g. meta-llama/llama-3.3-70b-instruct")
    p.add_argument(
        "--providers",
        help="Comma separated provider slugs to pin, e.g. together,deepinfra",
    )
    p.add_argument("--suite", default=DEFAULT_SUITE, help="Path to an eval suite JSON")
    p.add_argument("--repeats", type=int, default=1, help="Runs per case per provider")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run offline against bundled sample results, no API key required",
    )
    p.add_argument(
        "--list-providers",
        action="store_true",
        help="List provider slugs currently serving --model, then exit",
    )
    p.add_argument("--markdown", help="Write a Markdown report to this path")
    p.add_argument("--json-out", help="Write raw results JSON to this path")
    p.add_argument("--no-cost", action="store_true", help="Skip cost lookups")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI color")

    # Gate overrides
    p.add_argument("--max-ttft-p95", type=float, help="Gate: max TTFT p95 in ms")
    p.add_argument("--max-latency-p95", type=float, help="Gate: max total latency p95 in ms")
    p.add_argument("--max-error-rate", type=float, help="Gate: max error rate, 0-1")
    p.add_argument("--min-quality", type=float, help="Gate: min quality pass rate, 0-1")
    p.add_argument("--min-throughput", type=float, help="Gate: min completion throughput in tok/s")

    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be at least 1")

    gates = dict(DEFAULT_GATES)
    if args.max_ttft_p95 is not None:
        gates["max_ttft_p95_ms"] = args.max_ttft_p95
    if args.max_latency_p95 is not None:
        gates["max_latency_p95_ms"] = args.max_latency_p95
    if args.max_error_rate is not None:
        gates["max_error_rate"] = args.max_error_rate
    if args.min_quality is not None:
        gates["min_quality_pass_rate"] = args.min_quality
    if args.min_throughput is not None:
        gates["min_tokens_per_sec"] = args.min_throughput

    if args.max_error_rate is not None and not 0 <= args.max_error_rate <= 1:
        p.error("--max-error-rate must be between 0 and 1")
    if args.min_quality is not None and not 0 <= args.min_quality <= 1:
        p.error("--min-quality must be between 0 and 1")
    if any(value is not None and value < 0 for value in (
        args.max_ttft_p95, args.max_latency_p95, args.min_throughput,
    )):
        p.error("latency and throughput gates cannot be negative")

    # ---- offline demo ----------------------------------------------------
    if args.demo:
        with open(SAMPLE_RESULTS) as f:
            payload = json.load(f)
        results = dicts_to_results(payload["results"])
        model = payload.get("model", "meta-llama/llama-3.3-70b-instruct")
        print(
            f"\nOffline demo. {len(results)} recorded runs across "
            f"{len({r.provider for r in results})} providers for {model}."
        )
        reports = summarize(results, gates)
        print(terminal_report(reports, color=not args.no_color))

        if args.markdown:
            with open(args.markdown, "w") as f:
                f.write(markdown_report(reports, results, model))
            print(f"Markdown report written to {args.markdown}\n")
        return 0 if all(r.verdict == "GO" for r in reports) else 1

    # ---- live paths need a key -------------------------------------------
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print(
            "OPENROUTER_API_KEY is not set.\n"
            "Run 'python cli.py --demo' to see the harness work with no key.",
            file=sys.stderr,
        )
        return 2

    client = OpenRouterClient(api_key)

    if args.list_providers:
        if not args.model:
            print("--list-providers requires --model", file=sys.stderr)
            return 2
        slugs = client.list_endpoints(args.model)
        if not slugs:
            print(f"No endpoints found for {args.model}.")
            return 1
        print(f"\nProviders serving {args.model}:")
        for s in slugs:
            print(f"  {s}")
        print()
        return 0

    if not args.model or not args.providers:
        print("--model and --providers are required for a live run", file=sys.stderr)
        return 2

    providers = [s.strip() for s in args.providers.split(",") if s.strip()]
    if not providers:
        print("--providers must contain at least one provider slug", file=sys.stderr)
        return 2

    with open(args.suite) as f:
        suite = json.load(f)
    if not suite.get("cases"):
        print("The eval suite must contain at least one case", file=sys.stderr)
        return 2

    total = len(providers) * len(suite["cases"]) * args.repeats
    print(f"\nRunning {total} calls: {len(suite['cases'])} cases x "
          f"{len(providers)} providers x {args.repeats} repeats")
    print("  . pass    q quality fail    x request error")

    results = run_matrix(
        client,
        suite,
        args.model,
        providers,
        repeats=args.repeats,
        fetch_cost=not args.no_cost,
        on_progress=_progress,
    )

    reports = summarize(results, gates)
    print(terminal_report(reports, color=not args.no_color))

    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(markdown_report(reports, results, args.model))
        print(f"Markdown report written to {args.markdown}\n")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {"model": args.model, "results": results_to_dicts(results)}, f, indent=2
            )
        print(f"Raw results written to {args.json_out}\n")

    return 0 if all(r.verdict == "GO" for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
