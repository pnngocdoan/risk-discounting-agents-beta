#!/usr/bin/env bash
# Serve one model with vLLM as an OpenAI-style server, so many eval.py jobs can share the GPU.
# vLLM batches the concurrent requests itself, so the GPU stays busy.
#
# Usage (Kaggle, after `pip install vllm`):
#   bash scripts/30_serve_model.sh [hf_model_id] [served_name] [port]
#   bash scripts/30_serve_model.sh Qwen/Qwen2.5-3B-Instruct qwen2.5-3b 8000
# Then point the jobs at it:
#   export RDA_BASE_URL=http://127.0.0.1:8000/v1 RDA_API_KEY=none
#   and use --model api:qwen2.5-3b in scripts/31_run_parallel.py
# Stop it with:  pkill -f vllm.entrypoints.openai.api_server
#
# Settings (override with environment variables): MAX_LEN (default 12288), GPU_MEM (default 0.90),
# CUDA_VISIBLE_DEVICES (default 0, i.e. the first GPU only; the judge can use the other one).
set -euo pipefail
MODEL_ID=${1:-Qwen/Qwen2.5-3B-Instruct}
NAME=${2:-qwen2.5-3b}
PORT=${3:-8000}
MAX_LEN=${MAX_LEN:-12288}
GPU_MEM=${GPU_MEM:-0.90}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "a server is already answering on port $PORT"; exit 0
fi

# fp16 ("half") because the T4 has no bf16; tool calling is turned on for Qwen's tool format.
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" --served-model-name "$NAME" --dtype half \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_MEM" \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --port "$PORT" > vllm_server.log 2>&1 &
echo "starting $MODEL_ID as '$NAME' on port $PORT (log: vllm_server.log)"

for i in $(seq 1 120); do
  if curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "server is up after about $((i * 5)) s"; exit 0
  fi
  sleep 5
done
echo "server did not come up in 10 minutes; last lines of vllm_server.log:"; tail -20 vllm_server.log
exit 1
