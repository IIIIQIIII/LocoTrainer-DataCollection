#!/bin/bash
# Deploy 8 vLLM instances on 8x H100 GPUs for parallel trajectory collection

set -e

MODEL_PATH="LocoreMind/LocoTrainer-4B"  # HuggingFace model ID
BASE_PORT=8000

echo "🚀 Deploying 8 vLLM instances for LocoTrainer-4B..."
echo "Model: $MODEL_PATH"
echo "Base port: $BASE_PORT"
echo ""

# Kill any existing vLLM processes
echo "Stopping existing vLLM instances..."
pkill -f "vllm serve" || true
sleep 2

# Start 8 instances
for GPU_ID in {0..7}; do
    PORT=$((BASE_PORT + GPU_ID))
    echo "Starting vLLM on GPU $GPU_ID, port $PORT..."

    CUDA_VISIBLE_DEVICES=$GPU_ID nohup vllm serve $MODEL_PATH \
        --host 0.0.0.0 \
        --port $PORT \
        --max-model-len 131072 \
        --gpu-memory-utilization 0.95 \
        --dtype bfloat16 \
        --disable-log-requests \
        --trust-remote-code \
        > vllm_gpu${GPU_ID}.log 2>&1 &

    PID=$!
    echo "  ✓ GPU $GPU_ID started (PID: $PID, Port: $PORT)"
done

echo ""
echo "⏳ Waiting 30s for model loading..."
sleep 30

# Health check
echo ""
echo "🔍 Health check:"
for GPU_ID in {0..7}; do
    PORT=$((BASE_PORT + GPU_ID))
    if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
        echo "  ✓ GPU $GPU_ID (port $PORT): OK"
    else
        echo "  ✗ GPU $GPU_ID (port $PORT): FAILED"
    fi
done

echo ""
echo "✅ Deployment complete!"
echo "📊 Monitor GPUs: watch -n 1 nvidia-smi"
echo "📝 View logs: tail -f vllm_gpu0.log"
echo "🛑 Stop all: pkill -f 'vllm serve'"
