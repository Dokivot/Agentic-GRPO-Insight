# BASELINE.md — Baseline 规格定义

## Baseline：Qwen2.5-7B-Instruct 在 tau-bench-airline 上的零样本评测

最简单、最稳定、最可复现的 baseline。不训练、不微调，仅评测基础模型的开箱 agentic 能力。

## 模型

- 名称：Qwen2.5-7B-Instruct（HuggingFace: Qwen/Qwen2.5-7B-Instruct）
- 参数量：7.6B
- 上下文长度：32768 tokens
- 量化：无（BF16）

## 用户模拟器

- 名称：Qwen2.5-72B-AWQ（HuggingFace: Qwen/Qwen2.5-72B-Instruct-AWQ）
- 量化：AWQ 4-bit
- 部署：独立 vLLM 实例（TP=2，端口 8001）
- 温度：0.7（tau-bench 默认用户模拟温度）

## 推理（Agent）

- 引擎：vLLM
- Tensor parallel：TP=2
- GPU memory utilization：0.35（为 72B-AWQ 模拟器预留空间）
- Max model length：32768
- 服务方式：OpenAI 兼容 API，localhost:8000

## 推理（模拟器）

- 引擎：vLLM
- Tensor parallel：TP=2
- GPU memory utilization：0.50
- Max model length：8192（模拟器无需长上下文）
- 服务方式：OpenAI 兼容 API，localhost:8001

## 评测基准

- tau-bench-airline
- 版本：安装时锁定（记录确切版本号）
- 任务：全部 50 个 airline 任务
- 每任务 rollout 次数：1（贪心，temperature=0）
- 最大轮次：30（tau-bench 默认）

## 硬件

- 2x A800-80GB
- CUDA 12.1
- Docker 容器

## 收集指标

见计划中的"阶段指标规范 — 阶段 2：零样本 Baseline 评测"。

### 任务级
- total_tasks (50)、successful_tasks、task_success_rate
- sft_train_group (40 任务)：success_rate
- holdout_group (10 任务)：success_rate
- zero_success_tasks、per_task_results

### 轨迹级
- 轨迹长度（轮次）、上下文长度（每轮及累计）
- max/mean/P95/P99 上下文长度、输入/输出 token 数

### 工具使用
- 工具调用次数、有效率、JSON 解析失败数
- 无效工具名称、无效参数、执行失败数

### 基础设施
- 总评测时间、吞吐量、延迟、GPU 峰值内存、GPU 利用率

### 失效模式
- 每轨迹分类（见 Phase 6）

## 配置文件

`configs/baseline_eval.yaml` — 包含上述所有参数的冻结配置。

## 可复现性

可从以下要素复现：代码版本 + 配置 + 模型 ID + Docker 镜像 hash + 种子。
贪心解码 (temperature=0) 应产生确定性结果。

## 状态

Phase 4 — Baseline 定义：已完成
下一步：Phase 5 — Baseline 实现（代码）+ 云端执行
