# PROBLEM.md — Problem Definition

## Research Question

以 Qwen2.5-7B-Instruct 为基础模型，在 tau-bench-airline 多轮工具调用任务上使用 vanilla GRPO 进行 Agentic RL 训练时，主要失效模式是什么？能否通过有针对性的改进方法获得相对 vanilla baseline 的可量化、可复现提升？

## Motivation

GRPO (Group Relative Policy Optimization) 消除了 PPO 中的 value model，降低了复杂度和内存占用。但其在稀疏轨迹级奖励的 agentic 多轮场景中的行为尚不明确。tau-bench-airline (27 个工具、50 个任务、复杂多轮交互) 提供了一个失效模式丰富且可诊断的挑战性测试环境。

现有的 Agentic RL 研究多聚焦于单轮或少轮场景，对长轨迹、多工具、稀疏奖励条件下的 GRPO 行为缺乏系统性分析。本项目旨在填补这一空白，通过严格的实验流程识别并解决 vanilla GRPO 在多轮工具调用中的关键瓶颈。

## Objective

1. 建立可靠的 vanilla GRPO baseline (SFT 热启动 -> vanilla GRPO)
2. 系统性分析 tau-bench-airline 上的失效模式
3. 基于失效分析提出有针对性的改进方法
4. 通过受控实验验证改进是否有效
5. 形成相对 vanilla GRPO 有明确、可量化提升的最终方案

## Constraints

- 硬件：2x A800-80GB (共 160GB VRAM)
- 模型：Qwen2.5-7B-Instruct (基础模型)、Qwen2.5-72B-AWQ (教师模型 + 用户模拟器，全程自包含)
- 评测：tau-bench-airline 领域 (50 个任务，27 个工具)
- 训练：LoRA (内存约束 — 需在 2 块 GPU 上共存训练 + 参考模型 + rollout + 用户模拟器)
- 框架：veRL + vLLM 混合引擎
- 执行环境：云端服务器，Docker 部署
- 指标持久化：SwanLab + 本地 JSON 双通道，云端与本地双副本
- 所有结果必须可从 代码 + 配置 + 种子 复现，无外部 API 依赖

## Success Metrics

- 主要指标：任务成功率相对 vanilla GRPO baseline 有 >=5% 绝对提升
- 次要指标：已识别失效模式减少 (工具调用错误、上下文溢出、轨迹中止等)
- 第三指标：可复现性 (相同配置 -> 种子方差内相同结果)
- 所有改进须用 >=3 个随机种子验证

## Non-goals

- 修改模型架构 (不改层结构，不加新模块)
- 多领域泛化 (仅 airline)
- 超越 SOTA agent 系统
- 全参数微调 (仅 LoRA，受内存约束)
- 从零训练 (始终从 SFT 热启动)

## Assumptions

- ASSUMPTION: veRL 的多轮 GRPO 支持足够成熟，可用于 tau-bench 集成 (实现阶段验证)
- ASSUMPTION: Qwen2.5-72B-AWQ 在 airline 任务上能达到足够成功率以生成有意义的 SFT 数据 (若成功率过低，需更换教师模型或调整领域)
- ASSUMPTION: tau-bench 支持为 agent 和 user simulator 分别配置不同的 OpenAI 兼容 API 端点 (在 tau-bench 源码中验证)
- ASSUMPTION: 2x A800-80GB 足以同时运行 7B agent + 72B-AWQ simulator + GRPO 训练 (内存估算见 ARCHITECTURE.md，UNVERIFIED)
- ASSUMPTION: SwanLab 云端服务可正常访问 (若不可用，回退为纯本地 JSON + 本地 SwanLab 自托管)

## Unknowns

- veRL 多轮环境 API 的稳定性 (UNVERIFIED)
- Qwen2.5-72B-AWQ 在 airline 任务上的成功率 (NOT MEASURED)
- tau-bench 是否支持为 agent 和 user simulator 分别配置不同端点 (UNVERIFIED，需查源码)
- veRL 混合引擎能否与外部 vLLM 实例 (72B-AWQ simulator) 共存于同组 GPU (UNVERIFIED)
- 多轮 agentic 任务的 GRPO 最优超参数 (待调优)

## Status

Phase 1 — Problem Definition: COMPLETE
Next: Phase 2 — Technical Research
