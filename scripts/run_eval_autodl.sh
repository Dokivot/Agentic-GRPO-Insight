#!/usr/bin/env bash
# AutoDL 一键 baseline 评测脚本
# 启动 7B (GPU0) + 72B-AWQ (GPU1) vLLM → 运行评测 → 清理进程
#
# 用法:
#   bash scripts/run_eval_autodl.sh                          # 默认 baseline 评测
#   bash scripts/run_eval_autodl.sh --config configs/xxx.yaml # 指定配置
#   bash scripts/run_eval_autodl.sh --skip-vllm              # 跳过 vLLM 启动（已手动启动）
#   bash scripts/run_eval_autodl.sh --tiny                   # 快速 smoke test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# --- 配置 ---
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
CONDA_ENV_NAME="agentic-rl"
CONFIG_FILE="configs/baseline_eval.yaml"
SKIP_VLLM=false
TINY=false
VLLM_LOG_DIR="logs"

# --- 解析参数 ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"; shift 2 ;;
        --skip-vllm)
            SKIP_VLLM=true; shift ;;
        --tiny)
            TINY=true; shift ;;
        *)
            echo "未知参数: $1"; exit 1 ;;
    esac
done

mkdir -p "$VLLM_LOG_DIR"

# --- 激活 conda 环境 ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME" 2>/dev/null || {
    echo "警告: conda 环境 '$CONDA_ENV_NAME' 不存在，使用当前环境"
}

# --- 设置环境变量 ---
export VLLM_USE_V1=1
export HF_ENDPOINT="${HF_MIRROR:-https://hf-mirror.com}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy-key}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
export TOKENIZERS_PARALLELISM=false

# --- 健康检查函数 ---
wait_for_vllm() {
    local name=$1
    local port=$2
    local max_wait=${3:-300}
    local elapsed=0
    echo -n "  等待 $name (端口 $port) 启动..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -sf "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            echo " 就绪"
            return 0
        fi
        echo -n "."
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo " 超时（${max_wait}s）"
    return 1
}

# --- 启动 vLLM 服务 ---
VLLM_PIDS=()

if [ "$SKIP_VLLM" = false ]; then
    echo "============================================"
    echo "  启动 vLLM 服务"
    echo "============================================"

    # 7B Agent (GPU0, port 8000)
    if curl -sf "http://localhost:8000/v1/models" > /dev/null 2>&1; then
        echo "  [跳过] 端口 8000 已有服务在运行"
    else
        echo ""
        echo "[1/2] 启动 vLLM Agent (Qwen2.5-7B-Instruct, GPU0, 端口 8000)..."
        CUDA_VISIBLE_DEVICES=0 \
        MODEL_PATH="${AUTODL_TMP}/models/Qwen2.5-7B-Instruct" \
        PORT=8000 \
        TP_SIZE=1 \
        GPU_MEM_UTIL=0.80 \
        MAX_MODEL_LEN=16384 \
        CUDA_DEVICES=0 \
        bash scripts/vllm_server/7b.sh \
            > "$VLLM_LOG_DIR/vllm_agent.log" 2>&1 &
        VLLM_PIDS+=($!)
        echo "  PID: ${VLLM_PIDS[-1]}, 日志: $VLLM_LOG_DIR/vllm_agent.log"
    fi

    if ! wait_for_vllm "vLLM Agent" 8000 300; then
        echo "[错误] vLLM Agent 启动失败，查看日志: $VLLM_LOG_DIR/vllm_agent.log"
        tail -20 "$VLLM_LOG_DIR/vllm_agent.log" || true
        exit 1
    fi

    # 72B-AWQ Simulator (GPU1, port 8001)
    if curl -sf "http://localhost:8001/v1/models" > /dev/null 2>&1; then
        echo "  [跳过] 端口 8001 已有服务在运行"
    else
        echo ""
        echo "[2/2] 启动 vLLM Simulator (Qwen2.5-72B-AWQ, GPU1, 端口 8001)..."
        CUDA_VISIBLE_DEVICES=1 \
        MODEL_PATH="${AUTODL_TMP}/models/Qwen2.5-72B-Instruct-AWQ" \
        PORT=8001 \
        TP_SIZE=1 \
        GPU_MEM_UTIL=0.9 \
        MAX_MODEL_LEN=16384 \
        MAX_NUM_SEQS=8 \
        CUDA_DEVICES=1 \
        bash scripts/vllm_server/72b.sh \
            > "$VLLM_LOG_DIR/vllm_simulator.log" 2>&1 &
        VLLM_PIDS+=($!)
        echo "  PID: ${VLLM_PIDS[-1]}, 日志: $VLLM_LOG_DIR/vllm_simulator.log"
    fi

    if ! wait_for_vllm "vLLM Simulator" 8001 600; then
        echo "[错误] vLLM Simulator 启动失败，查看日志: $VLLM_LOG_DIR/vllm_simulator.log"
        tail -20 "$VLLM_LOG_DIR/vllm_simulator.log" || true
        exit 1
    fi

    echo ""
    echo "GPU 内存使用:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    echo ""
else
    echo "[跳过] vLLM 启动（--skip-vllm）"
    if ! curl -sf "http://localhost:8000/v1/models" > /dev/null 2>&1; then
        echo "[错误] 端口 8000 无服务，请先启动 vLLM 或去掉 --skip-vllm"
        exit 1
    fi
    if ! curl -sf "http://localhost:8001/v1/models" > /dev/null 2>&1; then
        echo "[错误] 端口 8001 无服务，请先启动 vLLM 或去掉 --skip-vllm"
        exit 1
    fi
    echo "  vLLM 服务已就绪"
fi

# --- 运行评测 ---
echo "============================================"
echo "  运行 tau-bench 评测"
echo "============================================"
echo "  配置: $CONFIG_FILE"
echo ""

TINY_FLAG=""
if [ "$TINY" = true ]; then
    TINY_FLAG="--tiny"
    echo "  [tiny 模式] 2 tasks x 2 samples"
fi

python scripts/eval/run_baseline_eval.py --config "$CONFIG_FILE" $TINY_FLAG
EVAL_EXIT_CODE=$?

echo ""
echo "============================================"
echo "  评测完成（退出码: $EVAL_EXIT_CODE）"
echo "============================================"

# --- 停止 vLLM ---
if [ ${#VLLM_PIDS[@]} -gt 0 ]; then
    echo ""
    echo "停止 vLLM 进程..."
    for pid in "${VLLM_PIDS[@]}"; do
        kill "$pid" 2>/dev/null && echo "  已停止 PID $pid" || echo "  PID $pid 已退出"
    done
    sleep 3
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    echo "vLLM 进程已停止"
fi

exit $EVAL_EXIT_CODE
