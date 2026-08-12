# EXPERIMENT_LOG.md — 实验历史记录

不得重写历史以使记录看起来更整洁。仅追加新发现。

---

## exp_001 — 零样本 Baseline 评测

**日期：** 2026-08-12（计划）

**假设：** 基础模型 (Qwen2.5-7B-Instruct) 因缺乏任务特定训练，在 tau-bench-airline 上成功率较低 (< 20%)。

**问题：** Qwen2.5-7B-Instruct 在 tau-bench-airline 上的开箱 agentic 能力如何？

**预期结果：** 低成功率，可识别的失效模式（工具调用错误、规划错误）。

**配置：**
- 模型：Qwen/Qwen2.5-7B-Instruct (BF16, 无量化)
- 模拟器：Qwen/Qwen2.5-72B-Instruct-AWQ (AWQ 4-bit, temp=0.7)
- 基准：tau-bench-airline，全部 50 任务
- 解码：贪心 (temperature=0)，每任务 1 rollout
- 最大轮次：30
- 硬件：2x A800-80GB
- 配置文件：configs/baseline_eval.yaml

**命令：**
```bash
docker compose -f docker/docker-compose.yml up vllm-agent vllm-simulator -d
docker compose -f docker/docker-compose.yml run eval
```

**状态：** 待执行（代码已实现，等待云端执行）

**结果：** 尚未运行

**分析：** 待定

---

## 时间线日志

（在下方追加新条目）

| 时间 | 事件 | 详情 |
|------|------|------|
| 2026-08-12 | 项目初始化 | 目录结构、研究文档、代码、Docker、配置已创建 |
| 2026-08-12 | exp_001 定义 | 零样本 baseline 评测，代码就绪，等待云端执行 |
