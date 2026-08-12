# Agentic RL: Vanilla GRPO on tau-bench-airline

Research project studying failure modes of vanilla GRPO in multi-turn tool-use scenarios.
Base model: Qwen2.5-7B-Instruct. Benchmark: tau-bench-airline (50 tasks, 27 tools).
Teacher/Simulator: Qwen2.5-72B-AWQ. Hardware: 2x A800-80GB.

## Quick Start (Cloud Server)

```bash
# 1. Clone and enter
git clone <repo-url> && cd DoProj

# 2. Build Docker image and download models
bash scripts/setup_env.sh

# 3. Set SwanLab API key (optional, falls back to local-only)
export SWANLAB_API_KEY=<your-key>

# 4. Start inference services (agent + simulator)
docker compose -f docker/docker-compose.yml up vllm-agent vllm-simulator -d

# 5. Run baseline evaluation
docker compose -f docker/docker-compose.yml run eval

# 6. Sync results back to local
bash scripts/sync_results.sh user@cloud-host
```

## Project Structure

```
research/        # Persistent project memory (PROBLEM, RESEARCH, ARCHITECTURE, etc.)
experiments/     # Per-experiment directories (exp_001/, exp_002/, ...)
results/         # Experiment results (metrics, trajectories, logs)
src/             # Source code (tau_bench, data, inference, training, analysis, utils)
configs/         # YAML configs for each experiment phase
scripts/         # Entry-point scripts (run_eval, setup_env, sync_results)
docker/          # Dockerfile and docker-compose.yml
```

## Experiment Workflow

1. Zero-shot baseline eval (exp_001) - no training
2. SFT warm-start (exp_002) - teacher trajectories via rejection sampling
3. Vanilla GRPO (exp_003) - the algorithm under study
4. Targeted improvements (exp_004+) - based on bottleneck analysis
5. Final validation - all tasks, multiple seeds

## Key Documents

- [research/PROBLEM.md](research/PROBLEM.md) - Research question and success criteria
- [research/RESEARCH.md](research/RESEARCH.md) - Technical literature review
- [research/ARCHITECTURE.md](research/ARCHITECTURE.md) - System design
- [research/BASELINE.md](research/BASELINE.md) - Baseline specification
- [research/DECISIONS.md](research/DECISIONS.md) - Technical decision records
- [research/EXPERIMENT_PLAN.md](research/EXPERIMENT_PLAN.md) - Proposed experiments
- [research/EXPERIMENT_LOG.md](research/EXPERIMENT_LOG.md) - Chronological experiment history

## Reproducibility

All results are reproducible from: code version + config + model ID + seed.
Metrics are persisted via SwanLab + local JSON (dual-channel, cloud + local copies).
