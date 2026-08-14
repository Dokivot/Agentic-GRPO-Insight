"""
Baseline evaluation entry point.

Uses the Python API approach (TauBenchWrapper + VLLMPolicy + run_eval)
instead of the broken CLI subprocess wrapper. Integrates with DoProj's
analysis infrastructure (MetricsProcessor, TrajectoryAnalyzer, BottleneckRanker,
MetricsRecorder).

Usage:
    python scripts/eval/run_baseline_eval.py --config configs/baseline_eval.yaml
    python scripts/eval/run_baseline_eval.py --config configs/baseline_eval.yaml --tiny
    python scripts/eval/run_baseline_eval.py --config configs/baseline_eval.yaml \
        --model-path /path/to/checkpoint
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging
import os
import time

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import yaml
import numpy as np

from src.envs.tau_bench_wrapper import TauBenchWrapper
from src.models.vllm_policy import VLLMPolicy
from src.evaluation.pass_k_eval import run_eval
from src.tau_bench.task_split import ensure_split_exists
from src.tau_bench.metrics import MetricsProcessor
from src.analysis.trajectory_analyzer import TrajectoryAnalyzer
from src.analysis.bottleneck_ranker import BottleneckRanker
from src.utils.metrics_recorder import MetricsRecorder
from src.utils.seed import set_seed
from src.utils.config import resolve_model_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("run_baseline_eval")


def main():
    parser = argparse.ArgumentParser(description="Run tau-bench baseline evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--model-path", default=None,
                        help="Override policy model path (for checkpoint eval)")
    parser.add_argument("--tiny", action="store_true",
                        help="Quick smoke test: 2 tasks x 2 samples")
    parser.add_argument("--skip-bottleneck", action="store_true",
                        help="Skip bottleneck analysis (for quick re-evals)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.tiny:
        cfg["eval"]["num_tasks"] = 2
        cfg["eval"]["num_samples_per_task"] = 2
        cfg["eval"]["num_workers"] = 2
        cfg["output"]["dir"] = cfg["output"]["dir"] + "_tiny"

    if args.model_path:
        cfg["policy"]["model_name"] = args.model_path
        logger.info("Overrode policy model: %s", args.model_path)

    exp_id = cfg.get("exp_id", "exp_001")
    phase = cfg.get("phase", "baseline_eval")
    seed = cfg.get("seed", 42)
    results_dir = Path(cfg["output"]["dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    logger.info("Ensuring task split exists (seed=%d)...", seed)
    task_split = ensure_split_exists(seed=seed)
    logger.info("Split: %d train, %d holdout",
                len(task_split.train_tasks), len(task_split.holdout_tasks))

    metrics_cfg = cfg.get("metrics", {})
    recorder = MetricsRecorder(
        exp_id=exp_id,
        phase=phase,
        results_dir=str(results_dir),
        swanlab_project=metrics_cfg.get("swanlab_project", "agentic-rl-tau-bench"),
        swanlab_run_name=metrics_cfg.get("swanlab_run_name", f"{exp_id}_{phase}"),
    )

    wrapper = TauBenchWrapper(
        env_name=cfg["env"]["name"],
        user_strategy=cfg["env"]["user_strategy"],
        user_model=cfg["env"]["user_model"],
        user_provider=cfg["env"]["user_provider"],
        user_base_url=cfg["env"].get("user_base_url"),
        task_split=cfg["env"]["task_split"],
    )

    shared_policy = VLLMPolicy(**cfg["policy"])

    def policy_factory():
        return shared_policy

    logger.info("Starting tau-bench evaluation...")
    eval_start = time.time()
    report = run_eval(
        wrapper=wrapper,
        policy_factory=policy_factory,
        num_tasks=cfg["eval"]["num_tasks"],
        num_samples_per_task=cfg["eval"]["num_samples_per_task"],
        max_turns=cfg["eval"]["max_turns"],
        num_workers=cfg["eval"]["num_workers"],
        output_dir=str(results_dir),
    )
    eval_elapsed = time.time() - eval_start
    logger.info("Evaluation completed in %.1f seconds", eval_elapsed)

    # Flatten per-task results into per-trajectory dicts for analysis
    all_trajectories = []
    for task_result in report.per_task_results:
        for traj_dict in task_result.get("trajectories", []):
            all_trajectories.append(traj_dict)

    logger.info("Processing %d trajectories through metrics pipeline...", len(all_trajectories))

    metrics_proc = MetricsProcessor(
        task_split=task_split,
        tokenizer_model=resolve_model_path(cfg["policy"]["model_name"]),
    )

    per_task = []
    for traj in all_trajectories:
        per_task.append(metrics_proc.process_trajectory(traj))

    summary = metrics_proc.compute_summary(
        per_task=per_task,
        eval_time_seconds=eval_elapsed,
    )

    # Failure analysis
    logger.info("Running failure analysis...")
    max_turns = cfg["eval"]["max_turns"]
    max_ctx = 16384  # vLLM max_model_len
    analyzer = TrajectoryAnalyzer(
        max_turns=max_turns,
        max_context_length=max_ctx,
    )
    failure_analysis = analyzer.analyze_all(all_trajectories)
    summary["failure_mode_distribution"] = failure_analysis.get("failure_mode_distribution", {})

    # Record metrics
    logger.info("Recording metrics (JSON + SwanLab)...")
    summary_record = {
        "exp_id": exp_id,
        "phase": phase,
        "task_success_rate": report.pass_at_1,
        "pass_hat_1": report.pass_hat_1,
        "pass_hat_4": report.pass_hat_4,
        "pass_hat_8": report.pass_hat_8,
        "avg_turns": report.avg_turns,
        "avg_tool_calls": report.avg_tool_calls,
        "error_rate": report.error_rate,
        "num_tasks": report.num_tasks,
        "num_samples_per_task": report.num_samples_per_task,
        "eval_time_seconds": eval_elapsed,
        **summary,
    }
    recorder.record_summary(summary_record)
    recorder.record_per_task(per_task)
    recorder.write_json(failure_analysis, "failure_analysis.json")

    # Bottleneck analysis
    if not args.skip_bottleneck:
        logger.info("Running bottleneck analysis...")
        ranker = BottleneckRanker(results_dir=str(results_dir))
        bottleneck_report = ranker.rank(
            failure_analysis=failure_analysis,
            summary_metrics=summary,
        )
        json_path, md_path = ranker.save_report(bottleneck_report)
        recorder.write_json(bottleneck_report, "bottleneck_report.json")
        logger.info("Bottleneck report: %s", md_path)

    # Write human-readable summary
    summary_md = results_dir / "summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(_format_summary(report, summary, failure_analysis, eval_elapsed))
    logger.info("Summary written to %s", summary_md)

    recorder.finish()
    logger.info("=== Evaluation complete ===")


def _format_summary(report, summary, failure_analysis, elapsed):
    lines = ["# Evaluation Summary\n"]
    lines.append(f"**pass@1 (any success):** {report.pass_at_1:.3f}")
    lines.append(f"**pass^1 (avg success):** {report.pass_hat_1:.3f}")
    lines.append(f"**pass^4 (stability):** {report.pass_hat_4:.3f}")
    lines.append(f"**pass^8 (stability):** {report.pass_hat_8:.3f}")
    lines.append(f"**Avg turns:** {report.avg_turns:.2f}")
    lines.append(f"**Avg tool calls:** {report.avg_tool_calls:.2f}")
    lines.append(f"**Error rate:** {report.error_rate:.3f}")
    lines.append(f"**Eval time:** {elapsed:.1f}s\n")

    lines.append("## Failure Mode Distribution\n")
    lines.append("| Mode | Count | Percentage |")
    lines.append("|------|-------|------------|")
    for mode, stats in summary.get("failure_mode_distribution", {}).items():
        if isinstance(stats, dict):
            lines.append(f"| {mode} | {stats.get('count', 0)} | {stats.get('pct', 0):.1f}% |")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
