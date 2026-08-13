#!/usr/bin/env bash
# 一键执行 baseline 评测：启动服务 → 等待就绪 → 运行评测 → 关闭服务 → 打印结果
#
# 用法（在云端服务器 ~/DoProj 目录下）：
#   bash scripts/run_baseline.sh
#
# 可选环境变量：
#   SWANLAB_API_KEY  — SwanLab API Key（不设则自动回退纯本地 JSON）
#   MODEL_CACHE      — 模型缓存路径（默认 /data/models）
#   SKIP_SETUP       — 设为 1 则跳过 setup_env.sh（镜像和模型已就绪时用）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

COMPOSE_FILE="docker/docker-compose.yml"
RESULTS_DIR="results/baseline"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()     { echo -e "${GREEN}[run_baseline]${NC} $1"; }
warn()    { echo -e "${YELLOW}[run_baseline]${NC} $1"; }
error()   { echo -e "${RED}[run_baseline]${NC} $1"; }

# ---------------------------------------------------------------- #
# 1. 环境搭建（可跳过）
# ---------------------------------------------------------------- #
if [ "${SKIP_SETUP:-0}" = "1" ]; then
    warn "跳过环境搭建（SKIP_SETUP=1）"
else
    log "步骤 1/6: 环境搭建（构建 Docker 镜像 + 下载模型）"
    bash scripts/setup_env.sh
fi

# ---------------------------------------------------------------- #
# 2. 启动推理服务
# ---------------------------------------------------------------- #
log "步骤 2/6: 启动 vLLM 推理服务（agent + simulator）"
docker compose -f "$COMPOSE_FILE" up vllm-agent vllm-simulator -d

# ---------------------------------------------------------------- #
# 3. 等待服务就绪
# ---------------------------------------------------------------- #
log "步骤 3/6: 等待服务就绪（72B 加载较慢，最长等待 10 分钟）"

wait_for_service() {
    local name=$1
    local port=$2
    local max_wait=600
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            log "  ${name} 就绪 (port ${port})"
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
        printf "\r  等待 %s ... %ds" "$name" "$elapsed"
    done
    echo ""
    error "${name} 在 ${max_wait}s 内未就绪"
    return 1
}

wait_for_service "vllm-agent"      8000
wait_for_service "vllm-simulator"  8001
echo ""

# ---------------------------------------------------------------- #
# 4. 运行评测
# ---------------------------------------------------------------- #
log "步骤 4/6: 运行 tau-bench baseline 评测"
docker compose -f "$COMPOSE_FILE" run --rm eval
EVAL_EXIT=$?

# ---------------------------------------------------------------- #
# 5. 关闭服务
# ---------------------------------------------------------------- #
log "步骤 5/6: 关闭推理服务"
docker compose -f "$COMPOSE_FILE" down

if [ $EVAL_EXIT -ne 0 ]; then
    error "评测失败（exit code ${EVAL_EXIT}），请查看 ${RESULTS_DIR}/eval.log"
    exit 1
fi

# ---------------------------------------------------------------- #
# 6. 打印结果摘要
# ---------------------------------------------------------------- #
log "步骤 6/6: 评测结果"
echo ""

if [ -f "${RESULTS_DIR}/summary.md" ]; then
    cat "${RESULTS_DIR}/summary.md"
else
    warn "summary.md 未找到，检查 metrics 目录："
    ls -la "${RESULTS_DIR}/metrics/" 2>/dev/null || warn "metrics 目录也不存在"
fi

echo ""
log "结果目录：${RESULTS_DIR}/"
log "指标文件：${RESULTS_DIR}/metrics/baseline_metrics.json"
log "评测日志：${RESULTS_DIR}/eval.log"
log ""
log "同步到本地：bash scripts/sync_results.sh user@<cloud-ip> --include-trajectories"
