# LLM Serving Benchmark

This benchmark compares Friendli Engine against an open-source inference engine (for example vLLM) through OpenAI-compatible `/v1/chat/completions` endpoints.

## Why these metrics

I selected the following metrics because they capture both user experience and engine efficiency:

- **TTFT (time to first token):** measures initial responsiveness and includes queueing effects.
- **End-to-end latency:** measures total time from request submission to final token.
- **Request throughput (req/s):** measures how many requests the engine completes per second.
- **Token throughput (tokens/s):** measures how efficiently the engine processes/generates tokens.

These are standard serving metrics used in LLM benchmarking and are especially useful because TTFT and latency degrade once an engine saturates under concurrency.

## Why this visualization

The benchmark generates a single **throughput vs p95 latency** graph.  
This is the clearest way to compare serving efficiency because it shows the trade-off frontier directly:

- Better engine = higher throughput at the same latency, or
- Better engine = lower latency at the same throughput.

This makes the performance gap easy to interpret for both technical and non-technical stakeholders.

## Fairness / reproducibility

The script keeps the following constant across both engines:

- Prompt set
- Prompt generation seed
- Output token cap
- Concurrency schedule
- Warmup policy
- Request transport (same HTTP client and code path)

Only the serving engine changes.

## Run

```bash
pip install httpx
python benchmark.py \
  --friendli-base-url http://FRIENDLI_HOST/v1 \
  --oss-base-url http://VLLM_HOST/v1 \
  --friendli-model YOUR_MODEL_NAME \
  --oss-model YOUR_MODEL_NAME \
  --requests-per-level 100 \
  --concurrency-levels 1,2,4,8,16,32 \
  --max-tokens 128
```

Outputs:

- `benchmark_results.json`
- `benchmark_results.csv`
- `benchmark_report.html`

Open `benchmark_report.html` in a browser to view the chart.
