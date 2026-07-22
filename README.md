# openrouter-provider-eval

A launch-gate harness for qualifying **providers**, not models.

OpenRouter routes every request through two independent decisions: which model
answers, and which provider serves that model. Most eval tooling only looks at
the first one. This looks at the second, because when you are onboarding a new
provider or approving one for a launch, the model is already chosen. The open
question is whether *this endpoint* can carry production traffic.

Every run ends in a **GO or NO-GO** per provider, with the reasons stated.

```
PROVIDER COMPARISON
================================================================================================
PROVIDER              VERDICT      ERR   QUAL   TTFT p50   TTFT p95    LAT p95   TOK/S  $/1K out
------------------------------------------------------------------------------------------------
fireworks             GO            0%   100%      337ms      383ms      912ms    96.2   $0.0011
together              GO            0%   100%      412ms      513ms    1,187ms    80.1   $0.0007
deepinfra/fp4         NO-GO         6%    65%      490ms      715ms    1,782ms    58.6   $0.0003
novita                NO-GO         6%   100%    3,607ms    4,399ms    6,364ms    29.2   $0.0005
------------------------------------------------------------------------------------------------

BLOCKERS
  deepinfra/fp4
    - error rate 5.6% exceeds 5%
    - quality pass rate 64.7% below 95%
  novita
    - error rate 5.6% exceeds 5%
    - TTFT p95 4399ms exceeds 3000ms

RECOMMENDATION
  Fastest passing provider: fireworks (TTFT p95 383ms)
  Cheapest passing provider: together ($0.0007 per 1K completion tokens)
  Suggested routing order: fireworks, together
```

That output is the point. The cheapest provider in the run is a quantized
endpoint that fails structured output and system-prompt adherence, and the
second cheapest is too slow to put in front of a user. Price alone would have
picked the wrong one twice.

## Run it

No dependencies. Python 3.10 or newer is required. No API key is needed to see
it work.

```bash
python3 cli.py --demo
```

That runs against bundled sample results so the harness is demonstrable in one
command. For a live run:

```bash
export OPENROUTER_API_KEY=sk-or-...

# Which providers currently serve this model?
python3 cli.py --model meta-llama/llama-3.3-70b-instruct --list-providers

# Qualify three of them
python3 cli.py --model meta-llama/llama-3.3-70b-instruct \
  --providers together,fireworks,deepinfra --repeats 3

# Produce a report for the launch ticket
python3 cli.py --model meta-llama/llama-3.3-70b-instruct \
  --providers together,fireworks --markdown launch-report.md
```

Tests:

```bash
python3 tests/test_evaluator.py
python3 tests/test_client.py
```

Exit code is 0 when every provider passes its gates and 1 otherwise, so this
drops into CI as a pre-launch check.

## How it measures

**Provider pinning.** Each call sets `provider.only` with
`allow_fallbacks: false`. Without disabling fallbacks the router is free to
serve the request from somewhere else, which silently invalidates the whole
comparison. This is the single easiest way to get a provider benchmark wrong.

**Time to first token, via streaming.** Total latency hides the difference
between a provider that starts responding in 300ms and one that stalls for two
seconds and then dumps the completion at once. For anything user-facing, TTFT
is the number that separates providers. Total latency and tokens/sec are
tracked too, but TTFT is the gate that matters most.

**Cost from the generation endpoint.** After each call, `GET /api/v1/generation?id=`
returns what was actually billed. Estimating cost from token counts and a
pricing table drifts, especially with caching discounts. Stats are not always
available the instant a stream closes, so the client retries with backoff
rather than reporting a null that would skew the average.

**Quality by assertion, not by model grading.** For provider qualification you
are not asking "is this a good model." You already picked the model. You are
asking "did this provider serve it correctly." So the suite checks the failures
that actually differ between endpoints: truncation, invalid JSON, dropped
system prompts, ignored length constraints. A quantized endpoint that is 3x
cheaper and quietly stops following instructions is the exact failure this is
built to catch.

## Gates

Defaults are deliberately conservative, and all are overridable per launch.

| Gate | Default |
|---|---|
| Max error rate | 5% |
| Min quality pass rate | 95% |
| Max TTFT p95 | 3,000ms |
| Max total latency p95 | 30,000ms |
| Min throughput | 5 tok/s |

```bash
python3 cli.py --demo --max-ttft-p95 5000 --max-latency-p95 30000 \
  --min-quality 0.90 --min-throughput 5
```

## Writing an eval suite

Suites are plain JSON. Each case is a prompt plus assertions.

```json
{
  "id": "structured_json",
  "prompt": "Return a JSON object with keys \"status\" and \"count\"...",
  "max_tokens": 128,
  "assertions": [
    { "type": "json_object", "value": { "status": "ok", "count": 3 } }
  ]
}
```

Assertion types: `contains`, `not_contains`, `regex`, `is_json`, `json_object`,
`min_length`, `max_length`. `json_object` parses the response and compares it
to the exact object in `value`. Multi-turn cases use a `messages` array instead
of `prompt`, which is how the system-prompt adherence case works.

`is_json` and `json_object` strip code fences before parsing. Models wrap JSON in fences
constantly even when told not to, and failing a provider for that would be
measuring the model's habit rather than the endpoint's correctness. Streaming
requests must also receive the `[DONE]` sentinel; partial streams and normalized
stream errors are recorded as request failures.

## Layout

```
cli.py                    Entry point, live and offline modes
src/client.py             OpenRouter client, stdlib only, streaming TTFT
src/evaluator.py          Assertions, aggregation, launch gates
src/report.py             Terminal table and Markdown report
evals/default_suite.json  Baseline provider qualification suite
samples/                  Recorded results for the keyless demo
tests/test_evaluator.py   Assertion, percentile, and gate tests
tests/test_client.py      Streaming and generation-metadata tests
```

## Notes

Built by [Alex Ovabor](https://github.com/aovabo). Provider slugs are the
lowercase tags OpenRouter shows on each model page, including variants like
`deepinfra/fp4`. Use `--list-providers` to pull the current list rather than
guessing, since a wrong slug can make every request fail and produce a misleading
qualification result.
