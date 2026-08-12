# DECISIONS.md — Technical Decision Records

## D001 — Inference Engine: vLLM

**Decision:** vLLM

**Alternatives:** vLLM vs SGLang

**Evidence:**
- vLLM: PagedAttention, broad framework support, native veRL integration, proven on tau-bench
- SGLang: RadixAttention, faster for some structured generation patterns

**Reason:** veRL integration is critical; SGLang lacks veRL support.

**Trade-offs:** SGLang may be faster for multi-turn prefix sharing (mitigated by vLLM's prefix caching).

**Risk:** Low. vLLM is widely used and well-tested.

**Fallback:** None needed — vLLM is the standard choice.

---

## D002 — Training Framework: veRL

**Decision:** veRL

**Alternatives:** veRL vs OpenRLHF vs TRL

**Evidence:**
- veRL: hybrid engine (co-locate training + rollout on same GPUs), native GRPO, multi-turn support, Ray-based
- OpenRLHF: Ray-based, PPO/GRPO, vLLM integration, but less optimized for multi-turn on limited GPUs
- TRL: simple, HF ecosystem, but limited multi-turn agentic RL support

**Reason:** Hybrid engine is essential for 2x A800 (can't afford separate training + rollout GPU groups).

**Trade-offs:** veRL is complex; may need custom tau-bench environment adapter. Mixed engine coexistence with external vLLM (72B simulator) on same GPUs needs verification.

**Risk:** Medium. veRL multi-turn API stability is UNVERIFIED.

**Fallback:** OpenRLHF if veRL multi-turn integration proves unstable.

---

## D003 — Fine-tuning Method: LoRA

**Decision:** LoRA (rank=64, alpha=128, dropout=0.05, target: all linear layers)

**Alternatives:** LoRA vs Full Fine-tuning

**Evidence (memory math, 2x A800-80GB, GRPO phase):**

| Component | Full FT (GB) | LoRA (GB) |
|-----------|-------------|-----------|
| 7B params (BF16) | 14 | 14 |
| Optimizer (AdamW) | 28 | 1 |
| Gradients | 14 | 0.25 |
| LoRA adapter | 0 | 0.5 |
| Reference model | 14 | 0 (shared base) |
| vLLM + KV cache | 34 | 39 |
| 72B-AWQ simulator | 20 | 20 |
| **Total** | **~124** | **~75** |
| **Per GPU (TP=2)** | **~62** | **~38** |
| **Headroom** | **~18** | **~42** |

**Reason:** Full FT leaves only ~18GB headroom with 72B simulator co-located. LoRA gives 42GB headroom.

**Trade-offs:** LoRA may limit achievable quality ceiling.

**Risk:** Low for memory. Medium for quality ceiling.

**Fallback:** Full FT if LoRA underperforms and memory allows (unlikely with 72B simulator co-located).

---

## D004 — RL Algorithm: GRPO (Given)

**Decision:** GRPO

**Reason:** Not a choice — the algorithm under study.

**GRPO vs PPO context:**
- GRPO advantages: no value model (saves ~14GB for 7B), fewer hyperparameters, simpler implementation
- GRPO risks: high variance with low success rates, no per-step value estimation

**Risk:** High variance when success rate is low (most groups have all-zero rewards -> std=0 -> no gradient).

---

## D005 — tau-bench Integration

**Decision:** Built-in eval for baseline, custom wrapper for GRPO

**Reason:**
- Baseline: tau-bench built-in `eval` command with vLLM as OpenAI-compatible backend (simplest, most reproducible)
- GRPO: custom tau-bench environment wrapper for veRL rollout engine (needed for multi-turn interaction during RL)

**Custom wrapper requirements:**
- Manage tau-bench environment state
- Handle tool execution
- Interface with user simulator
- Return trajectory + terminal reward

**Risk:** Medium. Custom wrapper is additional code surface that could have bugs.

---

## D006 — Evaluation Protocol

**Decision:** All 50 tasks, grouped reporting (sft_train/holdout)

**Protocol:**
- Baseline: greedy (temp=0), 1 rollout/task, all 50 tasks
- Iteration: temp=0.7, 3 rollouts/task, all 50 tasks
- Final: temp=0.7, 5 rollouts/task, all 50 tasks
- Always report sft_train (40 tasks) and holdout (10 tasks) separately

**Reason:** 50 tasks is small enough for full eval. Subset sampling unnecessary. Group reporting separates memorization from generalization.

**Risk:** None significant.

---

## D007 — SFT Data Pipeline

**Decision:** Qwen2.5-72B-AWQ teacher, all 50 tasks x 16 rollouts, 40 tasks' trajectories enter training set

**Sampling:** 4xgreedy + 4xT=0.5 + 4xT=0.8 + 4xT=1.0 = 16 rollouts/task -> 800 total trajectories

**Data partition:**
- 40 training tasks' successful trajectories -> SFT training set
- 10 holdout tasks' successful trajectories -> _holdout_reference.jsonl (reference only, never trained)
- Contaminated trajectories -> contaminated.jsonl (never trained)

**Reason:** Sampling all 50 tasks provides teacher reference on holdout tasks. Holdout trajectories don't constitute data leakage (agent training doesn't depend on them, eval runs independently).

**Risk:** Teacher success rate may be too low to generate sufficient data.

**Fallback:** If teacher success rate < 10%, consider alternative teacher or domain adjustment.

---

## D008 — User Simulator: Qwen2.5-72B-AWQ

**Decision:** Qwen2.5-72B-AWQ

**Alternatives:** gpt-4o-mini (tau-bench default) vs Qwen2.5-72B-AWQ (local)

**Reason:** Eliminates external API dependency, ensures reproducibility. 72B is large enough for high-quality user simulation.

**Trade-offs:**
- Pro: self-contained, reproducible, no API cost
- Con: simulator quality may differ from gpt-4o, results not directly comparable with external tau-bench studies

**Mitigation:** Record simulator model and config. All experiments use same simulator, ensuring valid relative comparisons.

**Architecture impact:** Eval and GRPO phases need 7B agent + 72B-AWQ simulator co-located (tighter memory budget).

---

## D009 — Docker Environment

**Decision:** Docker with CUDA 12.1 base image

**Base:** nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
**Stack:** Python 3.10, PyTorch 2.4.0+cu121, Flash Attention 2, vLLM, veRL, tau-bench, SwanLab

**Reason:** Full environment reproducibility. All versions pinned.

**Risk:** Low.

---

## D010 — Metrics Persistence: SwanLab + JSON Dual-Channel

**Decision:** SwanLab + local JSON dual-channel, cloud + local dual-copy

**Alternatives:** Pure SwanLab vs Pure local JSON vs SwanLab + JSON

**Reason:**
- JSON: authoritative structured record (never lost)
- SwanLab: training curve visualization and cross-experiment comparison
- Dual-copy: cloud JSON + rsync to local for backup

**Offline fallback:** If SwanLab cloud unreachable, auto-switch to local SwanLab self-hosted. JSON unaffected.

**Experiment identification:** exp_id (e.g., exp_001) as SwanLab project + run name, aligned with experiments/ directory and EXPERIMENT_LOG.md.

**Phase tagging:** Each phase tagged (sft_data_gen / baseline_eval / sft_train / grpo_train / eval / analysis) for SwanLab filtering.

## Status

Phase 2 — Decisions: COMPLETE
