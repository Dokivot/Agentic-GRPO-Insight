# RESEARCH.md — 技术调研

## 1. tau-bench 评测基准

### 概述

tau-bench 是一个用于评估多轮场景中工具-智能体-用户交互的基准。

- 来源：Yao et al., 2024（GitHub: sierra-research/tau-bench）
- 结构：task = (用户指令, 工具集, 数据库, 期望数据库状态)
- 评测方式：确定性数据库状态检查（非 LLM-judge）— 这是可靠性的关键优势
- 成功判定：二元（数据库状态是否匹配期望）

### 领域

| 领域 | 工具数 | 任务数 | 复杂度 |
|------|--------|--------|--------|
| retail | 9 | 115 | 较低 |
| airline | 27 | 50 | 较高 |
| telecom | 不定 | 不定 | 中等 |

本项目使用 **airline** 领域（27 个工具，50 个任务）。

### 用户模拟器

tau-bench 使用基于 LLM 的用户模拟器，遵循隐藏指令。
默认使用 gpt-4o-mini。本项目使用 Qwen2.5-72B-AWQ 作为自包含替代方案。

### 评测流程

```
任务 (指令 + 工具 + 数据库)
  -> 智能体生成动作 (工具调用)
  -> 用户模拟器响应
  -> 循环直到智能体宣布完成或达到最大轮次
  -> 检查最终数据库状态是否匹配期望
  -> 二元成功/失败
```

### 对 RL 的关键特性

- 仅终端奖励（稀疏）
- 多轮交互（变长轨迹）
- 工具调用必须是合法 JSON
- 上下文随轮次增长
- 确定性奖励（无需奖励模型）

---

## 2. GRPO 算法

### 来源

DeepSeekMath（Shao et al., 2024）。

### 核心机制

对每个 prompt 采样 G 个输出，计算组相对优势：

```
A_i = (r_i - mean(r)) / std(r)
```

无 value model — 比 PPO 简单，内存占用更少。

### KL 惩罚

对参考策略的 KL 散度惩罚（系数通常 0.04）。

### 多轮场景已知问题

1. 奖励稀疏：仅终端奖励，无中间信号
2. 长轨迹：更多 token，更多内存，更多计算
3. 信用分配：轨迹级奖励 -> 逐 token 信用分配不明确
4. 上下文增长：每轮输入增长 -> KV cache 压力，截断风险
5. 高方差：成功率低时，大多数组的奖励全零 -> std=0 -> 无梯度信号
6. 观测 mask：环境/工具响应的 token 不应参与策略梯度损失

---

## 3. 多轮 RL 挑战（文献）

### RAGEN

Agent RL 框架。关键洞见：观测 token 必须从策略梯度损失中 mask 掉。
仅智能体生成的 token 应接收梯度。

### ToRA / AgentTuning

工具集成推理。SFT + RL 流水线用于工具使用。
关键挑战：轨迹级奖励 -> 逐 token 信用分配。

### 常见模式

- RL 前的 SFT 热启动（几乎所有成功案例都使用 SFT 热启动）
- 轨迹过滤：SFT 仅在成功轨迹上训练
- RL 损失中的观测 mask
- 上下文长度管理对长轨迹至关重要

---

## 4. 框架

### veRL

- 混合引擎：训练 + rollout 共置在同一组 GPU
- 原生 GRPO 支持
- 多轮支持（通过环境适配器）
- 基于 Ray 的编排
- 对 2x A800 的关键优势：无法负担独立的训练 + rollout GPU 组

### OpenRLHF

- 基于 Ray，PPO/GRPO，vLLM 集成
- 在有限 GPU 上的多轮优化不足
- 回退方案：若 veRL 多轮集成不稳定则改用

### TRL

- 简单，HF 生态
- 多轮 agentic RL 支持有限
- 不适合本项目

### vLLM

- PagedAttention 高效管理 KV cache
- 框架支持广泛，原生 veRL 集成
- OpenAI 兼容 API，便于 tau-bench 集成
- Prefix caching 用于多轮（跨轮共享前缀）

### SGLang

- RadixAttention，部分结构化生成场景更快
- 缺乏 veRL 支持 — 本项目不可用

---

## 5. 模型

### Qwen2.5-7B-Instruct（基础模型）

- 7.6B 参数
- 上下文：32768 tokens
- 开箱即用的工具使用能力
- BF16：约 14GB VRAM

### Qwen2.5-72B-Instruct-AWQ（教师 + 模拟器）

- 72B 参数，AWQ 4-bit 量化
- AWQ：约 40GB VRAM（TP=2：约 20GB/GPU）
- 高质量工具使用和指令遵循
- 同时充当教师（SFT 数据生成）和用户模拟器

---

## 6. LoRA 微调

### 配置

- rank=64, alpha=128, dropout=0.05
- 目标：所有线性层（q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj）
- 可训练参数：7B 模型约 0.5GB
- 内存：基础模型 (14GB) + LoRA (0.5GB) + 优化器 (1GB) = 约 15.5GB

### 为何选择 LoRA 而非全参数微调

内存约束：2x A800-80GB 需共存训练 + rollout + 72B-AWQ 模拟器。
全参数微调会为 vLLM KV cache 和模拟器留下不足的内存。
详见 DECISIONS.md D003 的详细内存计算。

---

## 7. Agentic RL 已知失效模式

### 工具调用错误
- 工具调用输出中的 JSON 格式无效
- 错误的工具名称
- 错误的参数（类型不匹配、缺少必填字段）

### 上下文问题
- 长轨迹时上下文溢出
- 截断导致工具调用格式错误
- KV cache 耗尽

### 规划错误
- 过早终止（智能体过早宣布完成）
- 循环（智能体重复相同动作无进展）
- 错误动作序列（工具调用合法但策略不正确）

### 用户模拟器问题
- 模拟器放弃或行为异常
- 模拟器偏离隐藏指令

### RL 特有问题
- 全零奖励组（无梯度信号）
- 奖励黑客（智能体找到捷径）
- KL 坍缩（策略偏离参考过远）
- 训练不稳定（loss spike、NaN）

---

## 8. 硬件需求

### 2x A800-80GB 内存预算

| 阶段 | 每卡用量 | 余量 |
|------|---------|------|
| SFT 数据生成（仅 72B-AWQ） | 约 20GB + KV | 约 60GB |
| SFT 训练（7B + LoRA） | 约 16GB | 约 64GB |
| GRPO 训练（训练模式） | 约 33GB | 约 47GB |
| GRPO 训练（rollout 模式） | 约 62GB | 约 18GB |
| 评测（7B + 72B-AWQ 共存） | 约 57GB | 约 23GB |

### 瓶颈：GRPO Rollout 模式

每卡约 18GB 余量较紧张。缓解措施：
- 降低 72B-AWQ 模拟器的 gpu_memory_utilization
- 减少并发 rollout 数
- 降低 7B rollout 的 max_batch_tokens

## 状态

Phase 2 — 技术调研：已完成
下一步：Phase 3 — 系统架构
