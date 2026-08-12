# ARCHITECTURE.md — System Architecture

## System Overview

```
+-------------------------------------------------------------+
|                    Cloud Server (2x A800-80GB)              |
|                                                             |
|  +--------------------------------------------------------+ |
|  |  SFT Data Gen Phase (72B-AWQ only)                     | |
|  |  vLLM: Qwen2.5-72B-AWQ (TP=2, port 8000)              | |
|  |  Acts as: teacher agent + user simulator               | |
|  |  -> tau-bench-airline (all 50 tasks x 16 rollouts)     | |
|  |  -> trajectory collection -> success/contaminated filter| |
|  |  -> partition:                                         | |
|  |     40 train tasks success -> SFT training set          | |
|  |     10 holdout tasks success -> _holdout_reference.jsonl| |
|  |     contaminated -> contaminated.jsonl                  | |
|  |  -> MetricsRecorder -> JSON + SwanLab                  | |
|  +--------------------------------------------------------+ |
|                                                             |
|  +-----------+    +------------+    +---------------+      |
|  | SFT Train |    | GRPO Train |    | Eval (all)    |      |
|  | (LoRA)    |    | (LoRA,veRL)|    |               |      |
|  | HF Trainer|    | Hybrid Eng |    | vLLM agent    |      |
|  | DeepSpeed |    | (veRL)     |    | (7B, port8000)|      |
|  | ZeRO-2    |    | + tau-bench|    | vLLM sim      |      |
|  | 7B only   |    |   wrapper  |    | (72B-AWQ,8001)|      |
|  | (no 72B)  |    | + Rollout  |    | -> tau-bench  |      |
|  |           |    |   vLLM(7B) |    |   eval        |      |
|  +-----------+    | + ext vLLM |    +---------------+      |
|                   |   (72B sim)|                           |
|                   +------------+                           |
|                                                             |
|  All phases -> MetricsRecorder -> local JSON + SwanLab     |
|  All 50 tasks eval, report sft_train(40) / holdout(10)     |
+---------------------------+---------------------------------+
                            |
                scripts/sync_results.sh (rsync)
                            v
+-------------------------------------------------------------+
|                Local Repo (results/ copy)                   |
|  results/baseline/  results/sft_data_gen/                  |
|  results/sft_train/ results/grpo_train/                    |
|  results/analysis/  ...                                    |
+-------------------------------------------------------------+
```

## Data Flow

1. **Teacher rollout:** Qwen2.5-72B-AWQ -> vLLM (single instance, agent + simulator) -> tau-bench-airline (50 tasks x 16 rollouts) -> trajectory collection -> success/contaminated filter -> partition by task ownership (40 train -> SFT set; 10 holdout -> _holdout_reference.jsonl; contaminated -> contaminated.jsonl) -> MetricsRecorder
2. **SFT:** SFT training set -> train/val split (task-level, within 40 train tasks) -> HF Trainer + LoRA -> SFT model adapter -> MetricsRecorder (loss curve + post-train eval)
3. **GRPO:** SFT model -> veRL hybrid engine (7B rollout) + external vLLM (72B-AWQ simulator) -> tau-bench env wrapper -> rollout generation -> reward (task success) -> GRPO update -> MetricsRecorder (per-step metrics)
4. **Eval (any model):** vLLM agent (7B) + vLLM simulator (72B-AWQ) -> tau-bench eval -> raw trajectories -> metrics post-processor -> MetricsRecorder
5. **Analysis:** load all phase metrics -> trajectory analyzer -> bottleneck ranker -> MetricsRecorder
6. **Sync:** scripts/sync_results.sh rsync cloud results/ to local repo

## GPU Allocation

| Phase | GPU 0 | GPU 1 | Total/GPU |
|-------|-------|-------|-----------|
| SFT data gen | 72B-AWQ (TP=2) | 72B-AWQ (TP=2) | ~20GB + KV |
| SFT training | 7B + LoRA (TP=2) | 7B + LoRA (TP=2) | ~16GB |
| GRPO train (train mode) | 7B+LoRA+opt + 72B-AWQ | same | ~33GB |
| GRPO train (rollout mode) | 7B vLLM KV + 72B-AWQ | same | ~62GB |
| Eval | 7B agent + 72B-AWQ sim | same | ~57GB |

## Memory Budget

### Eval Phase (per GPU, TP=2, two vLLM instances)

| Component | Memory (GB) |
|-----------|------------|
| 7B agent weights (BF16, TP=2) | 7 |
| 72B-AWQ sim weights (4-bit, TP=2) | 20 |
| 7B agent KV cache | ~15 |
| 72B-AWQ sim KV cache | ~15 |
| **Total per GPU** | **~57** |
| **Available** | **80** |
| **Headroom** | **~23** |

### GRPO Phase (per GPU, TP=2, veRL hybrid + 72B-AWQ sim)

| Component | Memory (GB) | Train Mode | Rollout Mode |
|-----------|------------|------------|--------------|
| 7B weights (BF16, TP=2) | 7 | yes | yes |
| LoRA adapter + opt + grad | 1 | yes | - |
| Activations (batch=4, seq=8192) | 5 | yes | - |
| 7B vLLM KV cache (rollout) | 25 | - | yes |
| 72B-AWQ sim weights (4-bit, TP=2) | 20 | yes | yes |
| 72B-AWQ sim KV cache | 10 | - | yes |
| **Total per GPU** | | **~33** | **~62** |
| **Available** | | **80** | **80** |
| **Headroom** | | **~47** | **~18** |

Note: ~18GB headroom in rollout mode is tight. Mitigations: reduce 72B-AWQ sim gpu_memory_utilization, reduce concurrent rollouts, lower 7B rollout max_batch_tokens.

## Concurrency

- SFT data gen: all 50 tasks parallel (tau-bench supports concurrent execution, single vLLM instance)
- Eval: all 50 tasks parallel (two vLLM instances, ~30-50 concurrent requests)
- GRPO rollout: group size G=8, per-prompt batch, all rollouts via veRL vLLM parallel; 72B-AWQ sim handles corresponding user simulation requests

## Failure Boundaries

- vLLM OOM (dual instance) -> reduce gpu_memory_utilization or concurrent requests
- tau-bench timeout (max turns exceeded) -> record as trajectory failure, not crash
- User simulator failure -> tau-bench handles gracefully, records as task failure
- veRL training instability (loss spike, NaN) -> checkpoint rollback, reduce learning rate
- Context overflow (input > max_context_length) -> record as context overflow failure
- 72B-AWQ sim vs veRL GPU contention -> manage CUDA_VISIBLE_DEVICES or vLLM gpu_memory_utilization limits
- SwanLab network unreachable -> auto-fallback to pure local JSON, experiment unaffected

## Status

Phase 3 — Architecture: COMPLETE
Next: Phase 4 — Baseline
