# EXPERIMENT_LOG.md — Chronological Experiment History

Never rewrite history to make it look cleaner. Append new findings.

---

## exp_001 — Zero-shot Baseline Eval

**Date:** 2026-08-12 (planned)

**Hypothesis:** The base model (Qwen2.5-7B-Instruct) will have low success rate (< 20%) on tau-bench-airline due to lack of task-specific training.

**Question:** What is the out-of-box agentic capability of Qwen2.5-7B-Instruct on tau-bench-airline?

**Expected result:** Low success rate with identifiable failure patterns (tool-call errors, planning errors).

**Configuration:**
- Model: Qwen/Qwen2.5-7B-Instruct (BF16, no quantization)
- Simulator: Qwen/Qwen2.5-72B-Instruct-AWQ (AWQ 4-bit, temp=0.7)
- Benchmark: tau-bench-airline, all 50 tasks
- Decoding: greedy (temperature=0), 1 rollout/task
- Max turns: 30
- Hardware: 2x A800-80GB
- Config file: configs/baseline_eval.yaml

**Command:**
```bash
docker compose -f docker/docker-compose.yml up vllm-agent vllm-simulator -d
docker compose -f docker/docker-compose.yml run eval
```

**Status:** PENDING (code implemented, awaiting cloud execution)

**Results:** NOT YET RUN

**Analysis:** PENDING

---

## Chronological Log

(Append new entries below as experiments are executed)

| Timestamp | Event | Details |
|-----------|-------|---------|
| 2026-08-12 | Project initialized | Directory structure, research docs, code, Docker, configs created |
| 2026-08-12 | exp_001 defined | Zero-shot baseline eval, code ready, awaiting cloud execution |
