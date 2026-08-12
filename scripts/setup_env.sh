#!/usr/bin/env bash
# Cloud environment setup: build Docker, download models, verify GPU.
# Run on a fresh cloud instance with 2x A800-80GB.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================"
echo "  Agentic RL — Cloud Environment Setup"
echo "============================================"

# --- Verify GPU ---
echo ""
echo "[1/4] Verifying GPU..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "WARNING: Expected 2 GPUs, found $GPU_COUNT"
fi
echo "GPU check passed ($GPU_COUNT GPUs detected)."

# --- Build Docker image ---
echo ""
echo "[2/4] Building Docker image..."
docker build -f docker/Dockerfile -t agentic-rl:latest .
echo "Docker image built."

# --- Download models ---
echo ""
echo "[3/4] Downloading models..."
MODEL_CACHE="${MODEL_CACHE:-/data/models}"
mkdir -p "$MODEL_CACHE"

# Download Qwen2.5-7B-Instruct
if [ ! -d "$MODEL_CACHE/Qwen2.5-7B-Instruct" ]; then
    echo "  Downloading Qwen/Qwen2.5-7B-Instruct..."
    huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
        --local-dir "$MODEL_CACHE/Qwen2.5-7B-Instruct"
else
    echo "  Qwen2.5-7B-Instruct already cached."
fi

# Download Qwen2.5-72B-Instruct-AWQ
if [ ! -d "$MODEL_CACHE/Qwen2.5-72B-Instruct-AWQ" ]; then
    echo "  Downloading Qwen/Qwen2.5-72B-Instruct-AWQ..."
    huggingface-cli download Qwen/Qwen2.5-72B-Instruct-AWQ \
        --local-dir "$MODEL_CACHE/Qwen2.5-72B-Instruct-AWQ"
else
    echo "  Qwen2.5-72B-Instruct-AWQ already cached."
fi
echo "Models ready."

# --- Verify Docker compose ---
echo ""
echo "[4/4] Verifying Docker Compose config..."
docker compose -f docker/docker-compose.yml config --quiet
echo "Docker Compose config valid."

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. export SWANLAB_API_KEY=your_key"
echo "  2. docker compose -f docker/docker-compose.yml up vllm-agent vllm-simulator -d"
echo "  3. docker compose -f docker/docker-compose.yml run eval"
echo "  4. bash scripts/sync_results.sh user@cloud-host"
