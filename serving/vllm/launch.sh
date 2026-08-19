#!/usr/bin/env bash
: '
Serve Qwen3-8B-Instruct for the research agent on a single L40S (48 GB).

Every flag here is chosen for an AGENTIC workload: many short concurrent requests
with heavily repeated prefixes, not a few long-form generations.
'
set -euo pipefail

MODEL="${RESEARCH_MODEL:-Qwen/Qwen3-8B-Instruct}"

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name "$MODEL" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --disable-log-requests
