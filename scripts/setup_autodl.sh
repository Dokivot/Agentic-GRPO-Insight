#!/usr/bin/env bash
# AutoDL 环境配置脚本（无 Docker，conda + pip + patched tau-bench）
# 在全新 AutoDL 实例上运行：bash scripts/setup_autodl.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# AutoDL 数据盘路径
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
MODEL_CACHE="${MODEL_CACHE:-$AUTODL_TMP/models}"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"

echo "============================================"
echo "  DoProj — AutoDL 环境配置"
echo "============================================"
echo "  数据盘: $AUTODL_TMP"
echo "  模型缓存: $MODEL_CACHE"
echo "  pip 镜像: $PIP_MIRROR"
echo "  HF 镜像: $HF_MIRROR"
echo ""

# --- 验证 GPU ---
echo "[1/7] 验证 GPU..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "警告: 预期 2 块 GPU，检测到 $GPU_COUNT 块"
fi
echo "GPU 检查通过（$GPU_COUNT 块 GPU）"
echo ""

# --- 创建 conda 环境 ---
echo "[2/7] 创建 conda 环境..."
CONDA_ENV_NAME="agentic-rl"
if conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "  conda 环境 '$CONDA_ENV_NAME' 已存在，跳过创建"
else
    conda create -n "$CONDA_ENV_NAME" python=3.10 -y
    echo "  conda 环境 '$CONDA_ENV_NAME' 创建完成"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
echo "  当前 Python: $(python --version)"
echo ""

# --- 安装 PyTorch ---
echo "[3/7] 安装 PyTorch..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    echo "  PyTorch $TORCH_VER 已安装且 CUDA 可用，跳过安装"
else
    echo "  安装 PyTorch 2.7.0 + CUDA 12.6..."
    pip install torch==2.7.0 torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cu126
fi
echo ""

# --- 安装核心依赖 ---
echo "[4/7] 安装核心依赖..."
pip install -i "$PIP_MIRROR" --upgrade pip setuptools wheel
pip install -r "$PROJECT_ROOT/requirements.txt" -i "$PIP_MIRROR"

# Flash Attention（需从源码编译）
echo "  安装 Flash Attention（可能需要 5-10 分钟编译）..."
MAX_JOBS=4 pip install -i "$PIP_MIRROR" flash-attn --no-build-isolation || {
    echo "  [警告] Flash Attention 安装失败，vLLM 将使用替代 attention 后端"
}

# 安装本项目
echo "  安装本项目..."
pip install -i "$PIP_MIRROR" -e "$PROJECT_ROOT"
echo ""

# --- 安装 patched tau-bench ---
echo "[5/7] 安装 patched tau-bench..."
TAU_BENCH_DIR="$AUTODL_TMP/tau-bench"
if [ -d "$TAU_BENCH_DIR" ] && [ -f "$TAU_BENCH_DIR/setup.py" ]; then
    echo "  tau-bench 已克隆到 $TAU_BENCH_DIR"
else
    echo "  克隆 upstream tau-bench..."
    git clone https://github.com/sierra-research/tau-bench.git "$TAU_BENCH_DIR"
fi

echo "  应用 patches（添加 user_api_base 支持）..."
cp "$PROJECT_ROOT/third_party/tau_bench_patches/envs/__init__.py" "$TAU_BENCH_DIR/tau_bench/envs/__init__.py"
cp "$PROJECT_ROOT/third_party/tau_bench_patches/envs/base.py" "$TAU_BENCH_DIR/tau_bench/envs/base.py"
cp "$PROJECT_ROOT/third_party/tau_bench_patches/envs/user.py" "$TAU_BENCH_DIR/tau_bench/envs/user.py"
cp "$PROJECT_ROOT/third_party/tau_bench_patches/envs/airline/env.py" "$TAU_BENCH_DIR/tau_bench/envs/airline/env.py"
cp "$PROJECT_ROOT/third_party/tau_bench_patches/envs/retail/env.py" "$TAU_BENCH_DIR/tau_bench/envs/retail/env.py"

echo "  pip install -e tau-bench..."
pip install -i "$PIP_MIRROR" -e "$TAU_BENCH_DIR"
echo ""

# --- 下载模型 ---
echo "[6/7] 下载模型（使用 HF 镜像）..."
mkdir -p "$MODEL_CACHE"
export HF_ENDPOINT="$HF_MIRROR"

# Qwen2.5-7B-Instruct
MODEL_7B="$MODEL_CACHE/Qwen2.5-7B-Instruct"
if [ -d "$MODEL_7B" ] && [ -f "$MODEL_7B/config.json" ]; then
    echo "  Qwen2.5-7B-Instruct 已缓存"
else
    echo "  下载 Qwen/Qwen2.5-7B-Instruct..."
    huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
        --local-dir "$MODEL_7B"
fi

# Qwen2.5-72B-Instruct-AWQ
MODEL_72B="$MODEL_CACHE/Qwen2.5-72B-Instruct-AWQ"
if [ -d "$MODEL_72B" ] && [ -f "$MODEL_72B/config.json" ]; then
    echo "  Qwen2.5-72B-Instruct-AWQ 已缓存"
else
    echo "  下载 Qwen/Qwen2.5-72B-Instruct-AWQ（约 40GB，可能需要较长时间）..."
    huggingface-cli download Qwen/Qwen2.5-72B-Instruct-AWQ \
        --local-dir "$MODEL_72B"
fi
echo ""

# --- 生成任务切分 ---
echo "[7/7] 生成任务切分..."
python -m src.tau_bench.task_split || {
    echo "  [警告] 任务切分生成失败，将在评测时自动重试"
}
echo ""

# --- 验证安装 ---
echo "验证安装..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA 可用: {torch.cuda.is_available()}')
print(f'  GPU 数量: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1024**3:.1f} GB)')
"
python -c "import vllm; print(f'  vLLM: {vllm.__version__}')" 2>/dev/null || echo "  [警告] vLLM 导入失败"
python -c "from tau_bench.envs import get_env; print(f'  tau-bench: 已安装 (patched)')" 2>/dev/null || echo "  [警告] tau-bench 导入失败"
echo ""

echo "============================================"
echo "  AutoDL 环境配置完成！"
echo "============================================"
echo ""
echo "环境变量（建议加入 ~/.bashrc）:"
echo "  export AUTODL_TMP=$AUTODL_TMP"
echo "  export MODEL_CACHE=$MODEL_CACHE"
echo "  export HF_ENDPOINT=$HF_MIRROR"
echo "  export VLLM_USE_V1=1"
echo "  export SWANLAB_API_KEY=<your-key>"
echo ""
echo "下一步:"
echo "  1. conda activate $CONDA_ENV_NAME"
echo "  2. bash scripts/run_eval_autodl.sh           # 一键 baseline 评测"
echo "  3. 或手动启动 vLLM + 评测:"
echo "     bash scripts/vllm_server/7b.sh  &  (GPU0, port 8000)"
echo "     bash scripts/vllm_server/72b.sh &  (GPU1, port 8001)"
echo "     python scripts/eval/run_baseline_eval.py --config configs/baseline_eval.yaml"
