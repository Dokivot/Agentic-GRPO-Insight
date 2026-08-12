# Agentic RL：Vanilla GRPO 在 tau-bench-airline 上的失效模式研究

研究 vanilla GRPO 在多轮工具调用场景中的失效模式。基础模型：Qwen2.5-7B-Instruct。评测环境：tau-bench-airline（50 个任务，27 个工具）。教师/模拟器：Qwen2.5-72B-AWQ。硬件：2x A800-80GB。

## 快速开始（云端服务器）

```bash
# 1. 克隆并进入项目
git clone <repo-url> && cd DoProj

# 2. 构建 Docker 镜像并下载模型
bash scripts/setup_env.sh

# 3. 设置 SwanLab API Key（可选，不可用时自动回退为纯本地记录）
export SWANLAB_API_KEY=<your-key>

# 4. 启动推理服务（agent + simulator）
docker compose -f docker/docker-compose.yml up vllm-agent vllm-simulator -d

# 5. 运行 baseline 评测
docker compose -f docker/docker-compose.yml run eval

# 6. 同步结果到本地
bash scripts/sync_results.sh user@cloud-host
```

## 项目结构

```
research/        # 项目持久记忆（PROBLEM、RESEARCH、ARCHITECTURE 等）
experiments/     # 每个实验的独立目录（exp_001/、exp_002/、...）
results/         # 实验结果（指标、轨迹、日志）
src/             # 源代码（tau_bench、data、inference、training、analysis、utils）
configs/         # 各实验阶段的 YAML 配置
scripts/         # 入口脚本（run_eval、setup_env、sync_results）
docker/          # Dockerfile 和 docker-compose.yml
```

## 实验工作流

1. 零样本 baseline 评测 (exp_001) — 不训练
2. SFT 热启动 (exp_002) — 教师模型拒绝采样轨迹
3. Vanilla GRPO (exp_003) — 研究对象
4. 针对性改进 (exp_004+) — 基于瓶颈分析
5. 最终验证 — 全部任务，多种子

## 关键文档

- [research/PROBLEM.md](research/PROBLEM.md) — 研究问题与成功标准
- [research/RESEARCH.md](research/RESEARCH.md) — 技术文献调研
- [research/ARCHITECTURE.md](research/ARCHITECTURE.md) — 系统设计
- [research/BASELINE.md](research/BASELINE.md) — Baseline 规格定义
- [research/DECISIONS.md](research/DECISIONS.md) — 技术决策记录
- [research/EXPERIMENT_PLAN.md](research/EXPERIMENT_PLAN.md) — 拟议实验
- [research/EXPERIMENT_LOG.md](research/EXPERIMENT_LOG.md) — 实验历史记录

## 可复现性

所有结果可从以下要素复现：代码版本 + 配置 + 模型 ID + 种子。
指标通过 SwanLab + 本地 JSON 双通道持久化，云端与本地均保存副本。
