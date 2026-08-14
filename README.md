# Agentic RL：Vanilla GRPO 在 tau-bench-airline 上的失效模式研究

研究 vanilla GRPO 在多轮工具调用场景中的失效模式。基础模型：Qwen2.5-7B-Instruct。评测环境：tau-bench-airline（50 个任务，27 个工具）。教师/模拟器：Qwen2.5-72B-AWQ。硬件：2x A800-80GB。

## 快速开始（AutoDL）

```bash
# 1. 克隆并进入项目
git clone <repo-url> && cd DoProj

# 2. 一键环境配置（conda 环境 + 依赖 + patched tau-bench + 模型下载）
bash scripts/setup_autodl.sh

# 3. 激活环境
conda activate agentic-rl

# 4. 设置 SwanLab API Key（可选，不可用时自动回退为纯本地记录）
export SWANLAB_API_KEY=<your-key>

# 5. 一键 baseline 评测（自动启动 vLLM → 评测 → 清理）
bash scripts/run_eval_autodl.sh

# 或手动启动 vLLM + 评测：
bash scripts/vllm_server/7b.sh  &  # GPU0: 7B policy, port 8000
bash scripts/vllm_server/72b.sh &  # GPU1: 72B-AWQ user sim, port 8001
python scripts/eval/run_baseline_eval.py --config configs/baseline_eval.yaml
```

## SFT 管线

```bash
# 1. 数据采集（72B-AWQ 当 policy + user sim，best-of-16 拒绝采样）
bash scripts/vllm_server/72b.sh &  # GPU0: 72B policy, port 8000
# 需另开一个 72B 实例在 GPU1 port 8001 当 user sim
python scripts/train/sft/collect_sft_data.py --config configs/train/sft/sft_collect_airline.yaml

# 2. LoRA SFT 训练
python scripts/train/sft/sft_train.py --config configs/train/sft/sft_airline_lora.yaml

# 3. 合并 LoRA adapter
python scripts/train/sft/merge_lora.py \
    --base $AUTODL_TMP/models/Qwen2.5-7B-Instruct \
    --adapter experiments/sft_lora \
    --out $AUTODL_TMP/models/sft_lora_merged

# 4. SFT 评测
bash scripts/vllm_server/7b_sft.sh &  # GPU0: SFT-merged 7B, port 8000
bash scripts/vllm_server/72b.sh &     # GPU1: 72B-AWQ user sim, port 8001
python scripts/eval/eval_sft.py \
    --config configs/eval/eval_sft_airline.yaml \
    --split-file experiments/sft_collect_airline/split.json
```

## 项目结构

```
research/              # 项目持久记忆（PROBLEM、RESEARCH、ARCHITECTURE 等）
experiments/           # 实验输出目录
src/                   # 源代码
  envs/                # tau-bench 环境封装（TauBenchWrapper）
  models/              # vLLM 策略封装（VLLMPolicy）
  evaluation/          # pass^k 评测（run_eval）
  training/            # SFT 数据集和训练
  tau_bench/           # 任务切分和指标处理
  analysis/            # 失败分析和瓶颈排序
  utils/               # 配置加载、种子、指标记录
configs/               # 各实验阶段的 YAML 配置
scripts/               # 入口脚本
  eval/                # 评测入口（baseline + SFT）
  train/sft/           # SFT 管线（采集、训练、合并、检查）
  vllm_server/         # vLLM 服务启动脚本
third_party/           # patched tau-bench 文件（user_api_base 支持）
requirements.txt       # Python 依赖
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

## 实验产物保存

AutoDL 系统盘在实例关机后会被清空，只有数据盘 `/root/autodl-tmp` 持久保存。
`setup_autodl.sh` 会创建符号链接 `experiments -> /root/autodl-tmp/experiments`，
确保所有实验产物自动写入数据盘。

每步实验后运行 `save_artifacts.sh` 将产物分发到三处：

| 产物 | 大小 | 存储位置 | 方法 |
|------|------|---------|------|
| 评测摘要 (summary.md, metrics JSON) | 小 | GitHub | `save_artifacts.sh results --commit --push` |
| 瓶颈报告 (bottleneck_report) | 小 | GitHub | 同上 |
| 配置快照 (config.yaml) | 小 | GitHub | 同上 |
| 任务切分 (task_splits) | 小 | GitHub | 同上 |
| 完整评测报告 (eval_report.json) | 中-大 | HuggingFace Hub | `save_artifacts.sh hf` |
| SFT 数据集 (train.jsonl) | 大 | HuggingFace Hub | 同上 |
| LoRA adapter | 中 | HuggingFace Hub | 同上 |
| 合并后模型 (sft_lora_merged) | 大 | AutoDL 数据盘 | 已在 `/root/autodl-tmp/models/` |

```bash
# 保存小产物到 results/ 并推送到 GitHub
bash scripts/save_artifacts.sh results --commit --push

# 上传大产物到 HuggingFace Hub（需要 export HF_TOKEN=xxx）
bash scripts/save_artifacts.sh hf

# 全部保存
bash scripts/save_artifacts.sh all --commit --push
```

修改 `save_artifacts.sh` 顶部的 `HF_USER` 变量为你的 HuggingFace 用户名。
