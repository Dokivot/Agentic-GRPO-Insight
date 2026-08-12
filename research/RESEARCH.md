# RESEARCH.md — Technical Research

## 1. tau-bench Benchmark

### Overview

tau-bench is a benchmark for evaluating tool-agent-user interaction in multi-turn settings.

- Source: Yao et al., 2024 (GitHub: sierra-research/tau-bench)
- Structure: task = (user instruction, tool set, database, expected DB state)
- Evaluation: deterministic DB state check (not LLM-judge) — key reliability advantage
- Success: binary (DB state matches expected or not)

### Domains

| Domain | Tools | Tasks | Complexity |
|--------|-------|-------|------------|
| retail | 9 | 115 | Lower |
| airline | 27 | 50 | Higher |
| telecom | varies | varies | Medium |

This project uses **airline** (27 tools, 50 tasks).

### User Simulator

tau-bench uses an LLM-based user simulator that follows a hidden instruction.
Default: gpt-4o-mini. This project uses Qwen2.5-72B-AWQ as a self-contained alternative.

### Evaluation Flow

```
Task (instruction + tools + DB)
  -> Agent generates actions (tool calls)
  -> User simulator responds
  -> Repeat until agent declares done or max_turns
  -> Check final DB state against expected
  -> Binary success/fail
```

### Key Properties for RL

- Terminal reward only (sparse)
- Multi-turn (variable length trajectories)
- Tool calls must be valid JSON
- Context grows with each turn
- Deterministic reward (no reward model needed)

---

## 2. GRPO Algorithm

### Source

DeepSeekMath (Shao et al., 2024).

### Core Mechanism

For each prompt, sample G outputs. Compute group-relative advantage:

```
A_i = (r_i - mean(r)) / std(r)
```

No value model — simpler than PPO, less memory.

### KL Penalty

KL divergence penalty to reference policy (coefficient typically 0.04).

### Known Issues for Multi-turn

1. Reward sparsity: only terminal reward, no intermediate signal
2. Long trajectories: more tokens, more memory, more compute
3. Credit assignment: trajectory-level reward -> per-token credit is unclear
4. Context growth: input grows each turn -> KV cache pressure, truncation risk
5. High variance: when success rate is low, most groups have all-zero rewards -> std=0 -> no gradient signal
6. Observation masking: environment/tool response tokens should not contribute to policy gradient loss

---

## 3. Multi-turn RL Challenges (Literature)

### RAGEN

Agent RL framework. Key insight: observation tokens must be masked from policy gradient loss.
Only agent-generated tokens should receive gradient.

### ToRA / AgentTuning

Tool-integrated reasoning. SFT + RL pipeline for tool use.
Key challenge: trajectory-level reward -> per-token credit assignment.

### Common Patterns

- SFT warm-start before RL (几乎所有成功案例都使用 SFT 热启动)
- Trajectory filtering: only train on successful trajectories for SFT
- Observation masking in RL loss
- Context length management critical for long trajectories

---

## 4. Frameworks

### veRL

- Hybrid engine: training + rollout co-located on same GPUs
- Native GRPO support
- Multi-turn support (via environment adapter)
- Ray-based orchestration
- Key advantage for 2x A800: can't afford separate training + rollout GPU groups

### OpenRLHF

- Ray-based, PPO/GRPO, vLLM integration
- Less optimized for multi-turn on limited GPUs
- Fallback option if veRL multi-turn integration is unstable

### TRL

- Simple, HF ecosystem
- Limited multi-turn agentic RL support
- Not suitable for this project

### vLLM

- PagedAttention for efficient KV cache management
- Broad framework support, native veRL integration
- OpenAI-compatible API for tau-bench integration
- Prefix caching for multi-turn (shared prefix across turns)

### SGLang

- RadixAttention, faster for some structured generation
- Lacks veRL support — not viable for this project

---

## 5. Models

### Qwen2.5-7B-Instruct (Base Model)

- 7.6B parameters
- Context: 32768 tokens
- Good tool-use capability out of box
- BF16: ~14GB VRAM

### Qwen2.5-72B-Instruct-AWQ (Teacher + Simulator)

- 72B parameters, AWQ 4-bit quantized
- AWQ: ~40GB VRAM (TP=2: ~20GB/GPU)
- High-quality tool use and instruction following
- Serves as both teacher (SFT data generation) and user simulator

---

## 6. LoRA Fine-tuning

### Configuration

- rank=64, alpha=128, dropout=0.05
- Target: all linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- Trainable params: ~0.5GB for 7B model
- Memory: base model (14GB) + LoRA (0.5GB) + optimizer (1GB) = ~15.5GB

### Why LoRA over Full FT

Memory constraint: 2x A800-80GB must co-locate training + rollout + 72B-AWQ simulator.
Full FT would leave insufficient memory for vLLM KV cache and simulator.
See DECISIONS.md D003 for detailed memory math.

---

## 7. Known Failure Modes in Agentic RL

### Tool-call errors
- Invalid JSON in tool call output
- Wrong tool name
- Wrong arguments (type mismatch, missing required fields)

### Context issues
- Context overflow when trajectory is long
- Truncation leading to malformed tool calls
- KV cache exhaustion

### Planning errors
- Premature termination (agent declares done too early)
- Loops (agent repeats same action without progress)
- Wrong action sequence (valid tool calls but incorrect strategy)

### User simulator issues
- Simulator gives up or behaves unexpectedly
- Simulator deviates from hidden instruction

### RL-specific issues
- All-zero reward groups (no gradient signal)
- Reward hacking (agent finds shortcuts)
- KL collapse (policy diverges too far from reference)
- Training instability (loss spikes, NaN)

---

## 8. Hardware Requirements

### 2x A800-80GB Memory Budget

| Phase | Per-GPU Usage | Headroom |
|-------|--------------|----------|
| SFT data gen (72B-AWQ only) | ~20GB + KV | ~60GB |
| SFT training (7B + LoRA) | ~16GB | ~64GB |
| GRPO training (train mode) | ~33GB | ~47GB |
| GRPO training (rollout mode) | ~62GB | ~18GB |
| Eval (7B + 72B-AWQ co-located) | ~57GB | ~23GB |

### Bottleneck: GRPO Rollout Mode

~18GB headroom per GPU is tight. Mitigations:
- Reduce 72B-AWQ simulator gpu_memory_utilization
- Reduce concurrent rollout count
- Lower 7B rollout max_batch_tokens

## Status

Phase 2 — Technical Research: COMPLETE
Next: Phase 3 — Architecture
