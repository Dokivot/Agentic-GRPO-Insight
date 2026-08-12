# EXPERIMENT_PLAN.md — Proposed Experiments

## Overview

Experiments are proposed after baseline results are available and bottleneck analysis is complete.
Each experiment follows the lifecycle defined in AGENTS.md Section 4.

---

## exp_001 — Zero-shot Baseline Eval

- **Question:** What is the out-of-box agentic capability of Qwen2.5-7B-Instruct on tau-bench-airline?
- **Hypothesis:** The base model will have low success rate (< 20%) due to lack of task-specific training.
- **Motivation:** Establish the lowest baseline. All improvements measured against this.
- **Independent variables:** None (zero-shot, no training)
- **Controlled variables:** model, benchmark, max_turns, temperature (greedy), simulator model
- **Metrics:** task success rate, trajectory length, tool-call validity, context length, failure modes
- **Expected outcome:** Low success rate with identifiable failure patterns
- **Risk:** None
- **Estimated cost:** ~0.5 GPU-hours (eval only)
- **Status:** PENDING (code implemented, awaiting cloud execution)

---

## exp_002 — SFT Warm-start (Planned)

- **Question:** How much does SFT on teacher trajectories improve success rate over zero-shot?
- **Hypothesis:** SFT will significantly improve success rate and reduce tool-call errors.
- **Motivation:** Natural next step after zero-shot baseline. Required warm-start before GRPO.
- **Independent variables:** SFT training (LoRA on teacher trajectories)
- **Controlled variables:** base model, benchmark, eval protocol, simulator
- **Metrics:** task success rate (sft_train vs holdout), training loss, tool-call validity
- **Expected outcome:** Moderate success rate improvement, especially on sft_train tasks
- **Risk:** Overfitting to training tasks; low data quantity if teacher success rate is low
- **Estimated cost:** ~2 GPU-hours (data gen + SFT training + eval)
- **Status:** NOT STARTED (depends on exp_001 completion and bottleneck analysis)

---

## exp_003 — Vanilla GRPO (Planned)

- **Question:** What is the performance of vanilla GRPO after SFT warm-start?
- **Hypothesis:** GRPO will improve over SFT-only, but with identifiable failure modes.
- **Motivation:** This is the core baseline under study.
- **Independent variables:** GRPO training (LoRA, vanilla hyperparameters)
- **Controlled variables:** SFT model, benchmark, eval protocol, simulator
- **Metrics:** task success rate, reward curve, KL divergence, policy entropy, rollout success rate, failure modes
- **Expected outcome:** Improvement over SFT, but with clear bottlenecks
- **Risk:** Training instability, all-zero reward groups, OOM in rollout mode
- **Estimated cost:** ~8-16 GPU-hours (training + checkpoint evals)
- **Status:** NOT STARTED (depends on exp_002 completion)

---

## exp_004+ — Targeted Improvements (To Be Planned)

Proposed experiments based on bottleneck analysis of exp_003.
Each experiment should address one identified bottleneck with one primary hypothesis.

Template:
```
exp_NNN:
  question: ...
  hypothesis: ...
  motivation: ...
  independent_variables: ...
  controlled_variables: ...
  metrics: ...
  expected_outcome: ...
  risk: ...
  estimated_cost: ...
```

## Status

Phase 6 — Experiment Plan: INITIAL (exp_001-003 defined, exp_004+ pending bottleneck analysis)
