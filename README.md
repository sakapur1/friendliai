# LLM Serving Benchmark: Friendli Engine vs Open-Source Engine

This repository contains a reproducible API-level benchmark for comparing Friendli Engine with an open-source inference engine such as vLLM through OpenAI-compatible `/v1/chat/completions` endpoints.

## Overview

The benchmark is designed to support a deployment decision rather than a microbenchmarking exercise. It holds the prompt set, request shape, output length cap, warmup behavior, and concurrency schedule constant across both engines so that the measured differences primarily reflect serving efficiency rather than workload drift.

## Selected metrics

The benchmark measures four primary metrics because they jointly capture user experience and serving capacity.

- **TTFT (time to first token):** the initial responsiveness perceived by interactive users, including queueing delay and first-token generation delay.
- **End-to-end latency:** total request completion time from submission to final token, which matters for agent loops and longer completions.
- **Request throughput (req/s):** completed requests per second, which captures engine capacity at a given concurrency level.
- **Token throughput (tokens/s):** estimated input/output token processing rate, which is useful for comparing throughput under variable completion sizes.

These metrics are emphasized because queueing behavior under concurrency is one of the clearest indicators of inference efficiency in practice.

## Why this graph

The benchmark automatically generates a single **throughput vs p95 end-to-end latency** graph. This is the most decision-useful visualization because an engine is better if it achieves higher throughput at the same tail latency, or lower tail latency at the same throughput.

The chart also makes saturation behavior easy to see as concurrency increases. Tail latency is prioritized over average latency because p95 better captures the queuing and slowdown users notice once an engine approaches capacity.

## Fairness and reproducibility

The benchmark is API-level and intentionally avoids engine-internal instrumentation so that it can be run in a client-owned environment against already-deployed systems.To keep the comparison fair, the following are fixed across both engines:

- Prompt set and prompt generation seed.
- OpenAI-compatible `/chat/completions` request format.
- Output token cap (`max_tokens`).
- Concurrency schedule.
- Warmup policy before each measured round.
- Client transport path and measurement logic.

Only the serving engine changes.

## Repository contents

| File | Purpose |
|---|---|
| `benchmark.py` | Runs the benchmark against both endpoints and writes JSON, CSV, and HTML outputs. |
| `README.md` | Explains the benchmark design, metrics, and how to run it. |

## Requirements

- Python 3.10+.
- `httpx` installed for the async HTTP client.
- Two already-running OpenAI-compatible endpoints, one backed by Friendli Engine and one backed by the open-source engine.
- The same model family or an equivalently configured deployment on both sides for a fair comparison.

## Installation

```bash
pip install httpx
```

## Run

```bash
python benchmark.py \
  --friendli-base-url http://FRIENDLI_HOST/v1 \
  --oss-base-url http://VLLM_HOST/v1 \
  --friendli-model YOUR_MODEL_NAME \
  --oss-model YOUR_MODEL_NAME \
  --requests-per-level 100 \
  --concurrency-levels 1,2,4,8,16,32 \
  --max-tokens 128
```

## Outputs

After a run, the script writes three artifacts:

- `benchmark_results.json`: raw per-round summary metrics.
- `benchmark_results.csv`: tabular output for spreadsheets or notebooks.
- `benchmark_report.html`: a self-contained chart showing throughput vs p95 latency.

Open `benchmark_report.html` in a browser to review the graph.

## Interpretation

A stronger result for Friendli Engine appears when its curve sits to the right or below the open-source engine’s curve, meaning more throughput for the same p95 latency or lower p95 latency for the same throughput. If the curves cross, the benchmark still reveals the operating regions where each engine is more efficient, which is often more useful than a single-point comparison.

## Notes and limitations

This script estimates token counts from text length instead of using tokenizer-specific accounting, which keeps the benchmark portable but makes token-throughput numbers approximate. For a final customer-facing benchmark, tokenizer-accurate counts and a fixed prompt corpus matched to the target production workload would improve precision.

The benchmark is also API-level, so it captures client-visible performance rather than internal scheduler metrics. That is intentional because the client’s decision is about end-to-end serving efficiency of deployed systems, not just engine internals.
