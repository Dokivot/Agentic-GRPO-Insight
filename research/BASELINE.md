# BASELINE.md — Baseline Specification

## Baseline: Zero-shot Qwen2.5-7B-Instruct on tau-bench-airline

Simplest, most stable, most reproducible baseline. No training, no fine-tuning.
Evaluate the base model's out-of-box agentic capability.

## Model

- Name: Qwen2.5-7B-Instruct (HuggingFace: Qwen/Qwen2.5-7B-Instruct)
- Parameters: 7.6B
- Context length: 32768 tokens
- Quantization: None (BF16)

## User Simulator

- Name: Qwen2.5-72B-AWQ (HuggingFace: Qwen/Qwen2.5-72B-Instruct-AWQ)
- Quantization: AWQ 4-bit
- Deployment: independent vLLM instance (TP=2, port 8001)
- Temperature: 0.7 (tau-bench default user simulation temperature)

## Inference (Agent)

- Engine: vLLM
- Tensor parallel: TP=2
- GPU memory utilization: 0.35 (reserve space for 72B-AWQ simulator)
- Max model length: 32768
- Serving: OpenAI-compatible API on localhost:8000

## Inference (Simulator)

- Engine: vLLM
- Tensor parallel: TP=2
- GPU memory utilization: 0.50
- Max model length: 8192 (simulator does not need long context)
- Serving: OpenAI-compatible API on localhost:8001

## Benchmark

- tau-bench-airline
- Version: pinned at install time (record exact version)
- Tasks: all 50 airline tasks
- Rollouts per task: 1 (greedy, temperature=0)
- Max turns: 30 (tau-bench default)

## Hardware

- 2x A800-80GB
- CUDA 12.1
- Docker container

## Metrics Collected

See "Stage Metrics Spec — Stage 2: Zero-shot Baseline Eval" in plan.

### Task-level
- total_tasks (50), successful_tasks, task_success_rate
- sft_train_group (40 tasks): success_rate
- holdout_group (10 tasks): success_rate
- zero_success_tasks, per_task_results

### Trajectory-level
- trajectory length (turns), context length (per turn and cumulative)
- max/mean/P95/P99 context length, input/output tokens

### Tool-use
- tool call count, validity rate, JSON parse failures
- invalid tool names, invalid arguments, execution failures

### Infrastructure
- total eval time, throughput, latency, GPU peak memory, GPU utilization

### Failure modes
- classified per trajectory (see Phase 6)

## Config File

`configs/baseline_eval.yaml` — frozen config with all parameters above.

## Reproducibility

Reproducible from: code version + config + model ID + Docker image hash + seed.
Greedy decoding (temperature=0) should produce deterministic results.

## Status

Phase 4 — Baseline Definition: COMPLETE
Next: Phase 5 — Baseline Implementation (code) + cloud execution
