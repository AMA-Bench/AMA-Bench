#!/bin/bash
set -e

CONFIG="${1:-configs/qwen3-32B.yaml}"
[ ! -f "$CONFIG" ] && echo "Config not found: $CONFIG" && exit 1

# Parse all config at once
read -r MODEL VLLM_HOST VLLM_PORT GPUS MAX_LEN TP_SIZE < <(python -c "
import yaml
c = yaml.safe_load(open('$CONFIG'))
vl = c.get('vllm_launch', {})
print(c['model'], c.get('vllm_host', 'localhost'), c.get('vllm_port', 8000),
      vl.get('gpus', '0'), vl.get('max_model_len', 16384), vl.get('tensor_parallel_size', 1))
")

# Check if already running
curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null && \
    echo "vLLM already running at http://${VLLM_HOST}:${VLLM_PORT}" && exit 0

# Launch
echo "Launching vLLM: $MODEL on GPU $GPUS at http://${VLLM_HOST}:${VLLM_PORT}"
CUDA_VISIBLE_DEVICES=$GPUS nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host "$VLLM_HOST" --port "$VLLM_PORT" \
    --max-model-len "$MAX_LEN" --tensor-parallel-size "$TP_SIZE" \
    > vllm_server.log 2>&1 &

echo "PID: $! | Waiting for startup..."
for i in {1..60}; do
    curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null && echo "Ready!" && exit 0
    sleep 2
done

echo "Timeout. Check vllm_server.log" && exit 1
