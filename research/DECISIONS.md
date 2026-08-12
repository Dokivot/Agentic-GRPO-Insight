# DECISIONS.md — 技术决策记录

## D001 — 推理引擎：vLLM

**决策：** vLLM

**候选方案：** vLLM vs SGLang

**证据：**
- vLLM：PagedAttention，框架支持广泛，原生 veRL 集成，tau-bench 上已验证
- SGLang：RadixAttention，部分结构化生成场景更快

**理由：** veRL 集成是关键；SGLang 缺乏 veRL 支持。

**权衡：** SGLang 在多轮前缀共享上可能更快（vLLM 的 prefix caching 可缓解）。

**风险：** 低。vLLM 使用广泛且测试充分。

**回退方案：** 无需回退 — vLLM 是标准选择。

---

## D002 — 训练框架：veRL

**决策：** veRL

**候选方案：** veRL vs OpenRLHF vs TRL

**证据：**
- veRL：混合引擎（训练 + rollout 共置同组 GPU），原生 GRPO，多轮支持，基于 Ray
- OpenRLHF：基于 Ray，PPO/GRPO，vLLM 集成，但在有限 GPU 上多轮优化不足
- TRL：简单，HF 生态，但多轮 agentic RL 支持有限

**理由：** 混合引擎对 2x A800 至关重要（无法负担独立的训练 + rollout GPU 组）。

**权衡：** veRL 较复杂，可能需要自定义 tau-bench 环境适配器。混合引擎与外部 vLLM（72B 模拟器）在同组 GPU 上的共存需验证。

**风险：** 中等。veRL 多轮 API 稳定性为 UNVERIFIED。

**回退方案：** 若 veRL 多轮集成不稳定，改用 OpenRLHF。

---

## D003 — 微调方式：LoRA

**决策：** LoRA (rank=64, alpha=128, dropout=0.05, target: 所有线性层)

**候选方案：** LoRA vs 全参数微调

**证据（内存估算，2x A800-80GB，GRPO 阶段）：**

| 组件 | 全参数微调 (GB) | LoRA (GB) |
|------|----------------|-----------|
| 7B 参数 (BF16) | 14 | 14 |
| 优化器 (AdamW) | 28 | 1 |
| 梯度 | 14 | 0.25 |
| LoRA adapter | 0 | 0.5 |
| 参考模型 | 14 | 0 (共享 base) |
| vLLM + KV cache | 34 | 39 |
| 72B-AWQ 模拟器 | 20 | 20 |
| **总计** | **约 124** | **约 75** |
| **每卡 (TP=2)** | **约 62** | **约 38** |
| **余量** | **约 18** | **约 42** |

**理由：** 全参数微调与 72B 模拟器共存时仅剩约 18GB 余量。LoRA 留有 42GB 余量。

**权衡：** LoRA 可能限制可达到的质量上限。

**风险：** 内存风险低。质量上限风险中等。

**回退方案：** 若 LoRA 表现不佳且内存允许，改用全参数微调（但 72B 模拟器共存时几乎不可能）。

---

## D004 — RL 算法：GRPO（已确定）

**决策：** GRPO

**理由：** 非选择项 — 研究对象。

**GRPO vs PPO 背景：**
- GRPO 优势：无 value model（7B 省约 14GB），超参数更少，实现更简单
- GRPO 风险：成功率低时高方差，无逐步价值估计

**风险：** 成功率低时高方差（大多数组全零奖励 -> std=0 -> 无梯度）。

---

## D005 — tau-bench 集成

**决策：** Baseline 使用内置评测，GRPO 使用自定义封装

**理由：**
- Baseline：tau-bench 内置 `eval` 命令 + vLLM 作为 OpenAI 兼容后端（最简单、最可复现）
- GRPO：为 veRL rollout 引擎编写自定义 tau-bench 环境封装（多轮交互所需）

**自定义封装要求：**
- 管理 tau-bench 环境状态
- 处理工具执行
- 对接用户模拟器
- 返回轨迹 + 终端奖励

**风险：** 中等。自定义封装是额外的代码面，可能存在缺陷。

---

## D006 — 评测协议

**决策：** 全部 50 任务，分组报告 (sft_train/holdout)

**协议：**
- Baseline：贪心 (temp=0)，每任务 1 rollout，全部 50 任务
- 迭代评测：temp=0.7，每任务 3 rollout，全部 50 任务
- 最终评测：temp=0.7，每任务 5 rollout，全部 50 任务
- 始终分开报告 sft_train (40 任务) 和 holdout (10 任务)

**理由：** 50 个任务数量较少，无需子集采样。分组报告区分记忆与泛化。

**风险：** 无显著风险。

---

## D007 — SFT 数据流水线

**决策：** Qwen2.5-72B-AWQ 教师，全部 50 任务 x 16 rollout，40 任务的轨迹进入训练集

**采样：** 4xgreedy + 4xT=0.5 + 4xT=0.8 + 4xT=1.0 = 每任务 16 rollout -> 共 800 条轨迹

**数据划分：**
- 40 个训练任务的成功轨迹 -> SFT 训练集
- 10 个 holdout 任务的成功轨迹 -> _holdout_reference.jsonl（仅参考，永不训练）
- 污染轨迹 -> contaminated.jsonl（永不训练）

**理由：** 对全部 50 任务采样可获取教师在 holdout 任务上的参考表现。Holdout 轨迹不构成数据泄露（agent 训练不依赖这些轨迹，评测独立运行）。

**风险：** 教师成功率可能过低，导致数据量不足。

**回退方案：** 若教师成功率 < 10%，考虑更换教师模型或调整领域。

---

## D008 — 用户模拟器：Qwen2.5-72B-AWQ

**决策：** Qwen2.5-72B-AWQ

**候选方案：** gpt-4o-mini（tau-bench 默认）vs Qwen2.5-72B-AWQ（本地部署）

**理由：** 消除外部 API 依赖，保证可复现性。72B 参数量足以提供高质量用户模拟。

**权衡：**
- 优势：自包含、可复现、无 API 成本
- 劣势：模拟器质量可能与 gpt-4o 有差异，与外部 tau-bench 研究结果不可直接对比

**缓解：** 记录模拟器模型和配置。所有实验使用相同模拟器，确保相对比较的有效性。

**架构影响：** 评测和 GRPO 阶段需 7B agent + 72B-AWQ 模拟器共存（内存预算更紧张）。

---

## D009 — Docker 环境

**决策：** 基于 CUDA 12.1 的 Docker 镜像

**基础镜像：** nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
**软件栈：** Python 3.10, PyTorch 2.4.0+cu121, Flash Attention 2, vLLM, veRL, tau-bench, SwanLab

**理由：** 完整环境可复现。所有版本锁定。

**风险：** 低。

---

## D010 — 指标持久化：SwanLab + JSON 双通道

**决策：** SwanLab + 本地 JSON 双通道，云端 + 本地双副本

**候选方案：** 纯 SwanLab vs 纯本地 JSON vs SwanLab + JSON

**理由：**
- JSON：权威结构化记录（永不丢失）
- SwanLab：训练曲线可视化和跨实验对比
- 双副本：云端 JSON + rsync 到本地备份

**离线回退：** 若 SwanLab 云端不可达，自动切换为本地 SwanLab 自托管模式。JSON 不受影响。

**实验标识：** exp_id（如 exp_001）作为 SwanLab project + run name，与 experiments/ 目录和 EXPERIMENT_LOG.md 对齐。

**阶段标记：** 每个阶段用 phase tag 标记（sft_data_gen / baseline_eval / sft_train / grpo_train / eval / analysis），便于在 SwanLab 中按阶段筛选。

## 状态

Phase 2 — 技术决策：已完成
