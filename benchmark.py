#!/usr/bin/env python3
import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import httpx


@dataclass
class RequestResult:
    ok: bool
    status_code: int
    error: str | None
    input_tokens_est: int
    output_tokens_est: int
    ttft_ms: float | None
    e2e_ms: float | None


def percentile(values: List[float], p: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return vals[f]
    return vals[f] * (c - k) + vals[c] * (k - f)


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def build_prompts(n: int, seed: int) -> List[str]:
    random.seed(seed)
    base = [
        "Summarize the main point of this text in 3 bullet points: Async programming improves concurrency for I/O-bound workloads by allowing tasks to yield control during waits.",
        "Explain in 5 sentences the difference between throughput and latency in LLM inference.",
        "Write a concise code review comment for a Python function that catches broad exceptions and hides errors.",
        "Given a REST API that returns 429 errors under burst load, suggest 4 mitigations.",
        "Describe what TTFT measures and why it matters for interactive chat applications.",
    ]
    return [random.choice(base) for _ in range(n)]


async def one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
) -> RequestResult:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    start = time.perf_counter()
    first_token_at = None
    output_text_parts = []
    status_code = 0

    try:
        async with client.stream("POST", url, json=payload, timeout=timeout_s) as resp:
            status_code = resp.status_code
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = (
                        obj.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content")
                    )
                    if delta:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        output_text_parts.append(delta)

        end = time.perf_counter()
        output_text = "".join(output_text_parts)
        return RequestResult(
            ok=True,
            status_code=status_code,
            error=None,
            input_tokens_est=estimate_tokens(prompt),
            output_tokens_est=estimate_tokens(output_text) if output_text else 0,
            ttft_ms=((first_token_at - start) * 1000) if first_token_at else None,
            e2e_ms=(end - start) * 1000,
        )
    except Exception as e:
        end = time.perf_counter()
        return RequestResult(
            ok=False,
            status_code=status_code,
            error=str(e),
            input_tokens_est=estimate_tokens(prompt),
            output_tokens_est=0,
            ttft_ms=None,
            e2e_ms=(end - start) * 1000,
        )


async def run_round(
    name: str,
    base_url: str,
    model: str,
    prompts: List[str],
    concurrency: int,
    max_tokens: int,
    timeout_s: float,
) -> Dict[str, Any]:
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        sem = asyncio.Semaphore(concurrency)
        started = time.perf_counter()

        async def bounded(prompt: str):
            async with sem:
                return await one_request(client, base_url, model, prompt, max_tokens, timeout_s)

        results = await asyncio.gather(*[bounded(p) for p in prompts])
        duration = time.perf_counter() - started

    oks = [r for r in results if r.ok]
    ttfts = [r.ttft_ms for r in oks if r.ttft_ms is not None]
    e2es = [r.e2e_ms for r in oks if r.e2e_ms is not None]
    in_tok = sum(r.input_tokens_est for r in oks)
    out_tok = sum(r.output_tokens_est for r in oks)

    return {
        "engine": name,
        "concurrency": concurrency,
        "requests": len(results),
        "success": len(oks),
        "errors": len(results) - len(oks),
        "duration_s": duration,
        "req_throughput_rps": len(oks) / duration if duration else 0.0,
        "input_tok_per_s": in_tok / duration if duration else 0.0,
        "output_tok_per_s": out_tok / duration if duration else 0.0,
        "ttft_p50_ms": percentile(ttfts, 0.50),
        "ttft_p95_ms": percentile(ttfts, 0.95),
        "e2e_p50_ms": percentile(e2es, 0.50),
        "e2e_p95_ms": percentile(e2es, 0.95),
        "ttft_mean_ms": statistics.mean(ttfts) if ttfts else None,
        "e2e_mean_ms": statistics.mean(e2es) if e2es else None,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--friendli-base-url", required=True, help="e.g. http://friendli-host/v1")
    parser.add_argument("--oss-base-url", required=True, help="e.g. http://vllm-host/v1")
    parser.add_argument("--friendli-model", required=True)
    parser.add_argument("--oss-model", required=True)
    parser.add_argument("--requests-per-level", type=int, default=100)
    parser.add_argument("--concurrency-levels", default="1,2,4,8,16,32")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-json", default="benchmark_results.json")
    parser.add_argument("--out-csv", default="benchmark_results.csv")
    parser.add_argument("--out-html", default="benchmark_report.html")
    args = parser.parse_args()

    concurrency_levels = [int(x) for x in args.concurrency_levels.split(",")]
    all_rows = []

    for concurrency in concurrency_levels:
        prompts = build_prompts(args.requests_per_level, args.seed + concurrency)

        # Warmup
        warmup_prompts = prompts[:3]
        await run_round("friendli-warmup", args.friendli_base_url, args.friendli_model, warmup_prompts, min(2, concurrency), args.max_tokens, args.timeout_s)
        await run_round("oss-warmup", args.oss_base_url, args.oss_model, warmup_prompts, min(2, concurrency), args.max_tokens, args.timeout_s)

        friendli_row = await run_round(
            "Friendli Engine",
            args.friendli_base_url,
            args.friendli_model,
            prompts,
            concurrency,
            args.max_tokens,
            args.timeout_s,
        )
        oss_row = await run_round(
            "Open-source Engine",
            args.oss_base_url,
            args.oss_model,
            prompts,
            concurrency,
            args.max_tokens,
            args.timeout_s,
        )
        all_rows.extend([friendli_row, oss_row])

    with open(args.out_json, "w") as f:
        json.dump(all_rows, f, indent=2)

    csv_cols = list(all_rows[0].keys()) if all_rows else []
    with open(args.out_csv, "w") as f:
        f.write(",".join(csv_cols) + "\n")
        for row in all_rows:
            f.write(",".join("" if row[c] is None else str(row[c]) for c in csv_cols) + "\n")

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>LLM Serving Benchmark</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; }}
    #chart {{ width: 100%; height: 700px; }}
    table {{ border-collapse: collapse; margin-top: 24px; width: 100%; }}
    td, th {{ border: 1px solid #ccc; padding: 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Throughput vs p95 Latency</h1>
  <div id="chart"></div>
  <script>
    const rows = {json.dumps(all_rows)};
    const friendli = rows.filter(r => r.engine === "Friendli Engine");
    const oss = rows.filter(r => r.engine === "Open-source Engine");

    const traces = [
      {{
        x: friendli.map(r => r.req_throughput_rps),
        y: friendli.map(r => r.e2e_p95_ms),
        text: friendli.map(r => "conc=" + r.concurrency),
        mode: "lines+markers+text",
        textposition: "top center",
        name: "Friendli Engine"
      }},
      {{
        x: oss.map(r => r.req_throughput_rps),
        y: oss.map(r => r.e2e_p95_ms),
        text: oss.map(r => "conc=" + r.concurrency),
        mode: "lines+markers+text",
        textposition: "top center",
        name: "Open-source Engine"
      }}
    ];

    Plotly.newPlot("chart", traces, {{
      xaxis: {{ title: "Request throughput (req/s)" }},
      yaxis: {{ title: "p95 end-to-end latency (ms)" }},
      title: "Serving efficiency frontier"
    }});
  </script>
</body>
</html>
"""
    with open(args.out_html, "w") as f:
        f.write(html)

    print(f"Wrote {args.out_json}, {args.out_csv}, {args.out_html}")


if __name__ == "__main__":
    asyncio.run(main())
