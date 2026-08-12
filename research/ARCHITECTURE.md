# ARCHITECTURE.md — 系统架构

## 系统总览

```
+-------------------------------------------------------------+
|                    云端服务器 (2x A800-80GB)                 |
|                                                             |
|  +--------------------------------------------------------+ |
|  |  SFT 数据生成阶段 (仅需 72B-AWQ)                        | |
|  |  vLLM: Qwen2.5-72B-AWQ (TP=2, 端口 8000)               | |
|  |  同时充当: teacher agent + user simulator              | |
|  |  -> tau-bench-airline (全部 50 任务 x 16 rollout)      | |
|  |  -> 轨迹收集 -> 成功/污染过滤                           | |
|  |  -> 按任务归属划分:                                     | |
|  |     40 训练任务成功 -> SFT 训练集                       | |
|  |     10 holdout 任务成功 -> _holdout_reference.jsonl    | |
|  |     污染 -> contaminated.jsonl                         | |
|  |  -> MetricsRecorder -> JSON + SwanLab                  | |
|  +--------------------------------------------------------+ |
|                                                             |
|  +-----------+    +------------+    +---------------+      |
|  | SFT 训练  |    | GRPO 训练  |    | 评测 (所有阶段)|     |
|  | (LoRA)    |    | (LoRA,veRL)|    |               |      |
|  | HF Trainer|    | 混合引擎   |    | vLLM agent    |      |
|  | DeepSpeed |    | (veRL)     |    | (7B, 端口8000)|      |
|  | ZeRO-2    |    | + tau-bench|    | vLLM 模拟器   |      |
|  | 仅需 7B   |    |   环境封装 |    | (72B-AWQ,8001)|      |
|  | (无 72B)  |    | + Rollout  |    | -> tau-bench  |      |
|  |           |    |   vLLM(7B) |    |   评测        |      |
|  +-----------+    | + 外部 vLLM|    +---------------+      |
|                   |  (72B 模拟)|                           |
|                   +------------+                           |
|                                                             |
|  所有阶段 -> MetricsRecorder -> 本地 JSON + SwanLab        |
|  全部 50 任务评测，按 sft_train(40) / holdout(10) 分组报告  |
+---------------------------+---------------------------------+
                            |
                scripts/sync_results.sh (rsync)
                            v
+-------------------------------------------------------------+
|                    本地仓库 (results/ 副本)                 |
|  results/baseline/  results/sft_data_gen/                  |
|  results/sft_train/ results/grpo_train/                    |
|  results/analysis/  ...                                    |
+-------------------------------------------------------------+
```

## 数据流

1. **教师 rollout：** Qwen2.5-72B-AWQ -> vLLM（单实例，同时充当 agent 和 simulator）-> tau-bench-airline（全部 50 任务 x 16 rollout）-> 轨迹收集 -> 成功/污染过滤 -> 按任务归属划分（40 训练任务成功 -> SFT 训练集；10 holdout 任务成功 -> _holdout_reference.jsonl；污染 -> contaminated.jsonl）-> MetricsRecorder
2. **SFT：** SFT 训练集 -> 训练/验证切分（任务级，从 40 个训练任务内部切分）-> HF Trainer + LoRA -> SFT 模型 adapter -> MetricsRecorder（loss 曲线 + 训练后评测）
3. **GRPO：** SFT 模型 -> veRL 混合引擎（7B rollout）+ 外部 vLLM（72B-AWQ 模拟器）-> tau-bench 环境封装 -> rollout 生成 -> 奖励（任务成功）-> GRPO 更新 -> MetricsRecorder（每步指标）
4. **评测（任意模型）：** vLLM agent（7B）+ vLLM 模拟器（72B-AWQ）-> tau-bench 评测 -> 原始轨迹 -> 指标后处理 -> MetricsRecorder
5. **分析：** 加载所有阶段指标 -> 轨迹分析器 -> 瓶颈排序器 -> MetricsRecorder
6. **同步：** scripts/sync_results.sh 将云端 results/ rsync 到本地仓库

## GPU 分配

| 阶段 | GPU 0 | GPU 1 | 每卡总计 |
|------|-------|-------|---------|
| SFT 数据生成 | 72B-AWQ (TP=2) | 72B-AWQ (TP=2) | 约 20GB + KV |
| SFT 训练 | 7B + LoRA (TP=2) | 7B + LoRA (TP=2) | 约 16GB |
| GRPO 训练（训练模式） | 7B+LoRA+优化器 + 72B-AWQ | 同左 | 约 33GB |
| GRPO 训练（rollout 模式） | 7B vLLM KV + 72B-AWQ | 同左 | 约 62GB |
| 评测 | 7B agent + 72B-AWQ 模拟器 | 同左 | 约 57GB |

## 内存预算

### 评测阶段（每卡，TP=2，两个 vLLM 实例共存）

| 组件 | 内存 (GB) |
|------|----------|
| 7B agent 权重 (BF16, TP=2) | 7 |
| 72B-AWQ 模拟器权重 (4-bit, TP=2) | 20 |
| 7B agent KV cache | 约 15 |
| 72B-AWQ 模拟器 KV cache | 约 15 |
| **每卡总计** | **约 57** |
| **可用** | **80** |
| **余量** | **约 23** |

### GRPO 阶段（每卡，TP=2，veRL 混合引擎 + 72B-AWQ 模拟器）

| 组件 | 内存 (GB) | 训练模式 | Rollout 模式 |
|------|----------|---------|-------------|
| 7B 权重 (BF16, TP=2) | 7 | 是 | 是 |
| LoRA adapter + 优化器 + 梯度 | 1 | 是 | - |
| 激活值 (batch=4, seq=8192) | 5 | 是 | - |
| 7B vLLM KV cache (rollout) | 25 | - | 是 |
| 72B-AWQ 模拟器权重 (4-bit, TP=2) | 20 | 是 | 是 |
| 72B-AWQ 模拟器 KV cache | 10 | - | 是 |
| **每卡总计** | | **约 33** | **约 62** |
| **可用** | | **80** | **80** |
| **余量** | | **约 47** | **约 18** |

注：rollout 模式下每卡约 18GB 余量较紧张。缓解措施：降低 72B-AWQ 模拟器的 gpu_memory_utilization、减少并发 rollout 数、降低 7B rollout 的 max_batch_tokens。

## 并发

- SFT 数据生成：全部 50 任务并行（tau-bench 支持并发执行，单 vLLM 实例）
- 评测：全部 50 任务并行（两个 vLLM 实例，约 30-50 并发请求）
- GRPO rollout：组大小 G=8，按 prompt batch，所有 rollout 通过 veRL vLLM 并行；72B-AWQ 模拟器同时处理对应用户模拟请求

## 故障边界

- vLLM OOM（双实例共存）-> 降低 gpu_memory_utilization 或减少并发请求
- tau-bench 超时（超过最大轮次）-> 记录为轨迹失败，非崩溃
- 用户模拟器故障 -> tau-bench 优雅处理，记录为任务失败
- veRL 训练不稳定（loss spike、NaN）-> checkpoint 回滚，降低学习率
- 上下文溢出（input > max_context_length）-> 记录为上下文溢出失败
- 72B-AWQ 模拟器与 veRL 混合引擎 GPU 竞争 -> 管理 CUDA_VISIBLE_DEVICES 或 vLLM gpu_memory_utilization 限制
- SwanLab 网络不可达 -> 自动回退为纯本地 JSON 记录，实验不受影响

## 状态

Phase 3 — 系统架构：已完成
下一步：Phase 4 — Baseline
