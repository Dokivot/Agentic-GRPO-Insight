#!/usr/bin/env python3
"""Evaluation entry point.

Loads config, waits for vLLM services, runs tau-bench evaluation,
collects metrics, runs failure analysis, and saves all results via
the dual-channel MetricsRecorder (JSON + SwanLab).

Usage:
    python scripts/run_eval.py --config configs/baseline_eval.yaml
    python scripts/run_eval.py --config configs/baseline_eval.yaml \
        --model-path /path/to/checkpoint --num-rollouts 3 --temperature 0.7
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, save_config
from src.utils.seed import set_seed
from src.utils.metrics_recorder import MetricsRecorder
from src.tau_bench.task_split import ensure_split_exists
from src.tau_bench.evaluator import TauBenchEvaluator
from src.tau_bench.metrics import MetricsProcessor
from src.analysis.trajectory_analyzer import TrajectoryAnalyzer
from src.analysis.bottleneck_ranker import BottleneckRanker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("run_eval")


def main():
    parser = argparse.ArgumentParser(description="Run tau-bench evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--model-path", default=None,
                        help="Override agent model path (for checkpoint eval)")
    parser.add_argument("--num-rollouts", type=int, default=None,
                        help="Override number of rollouts per task")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override agent temperature")
    parser.add_argument("--skip-bottleneck", action="store_true",
                        help="Skip bottleneck analysis (for quick re-evals)")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    #  Load and optionally override config                               #
    # ------------------------------------------------------------------ #
    config = load_config(args.config)

    if args.model_path:
        config["agent_model"]["name"] = args.model_path
        logger.info("Overrode agent model: %s", args.model_path)
    if args.num_rollouts is not None:
        config["tau_bench"]["num_rollouts"] = args.num_rollouts
    if args.temperature is not None:
        config["tau_bench"]["agent_temperature"] = args.temperature

    exp_id = config.get("exp_id", "exp_001")
    phase = config.get("phase", "baseline_eval")
    seed = config.get("seed", 42)
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    # Freeze config snapshot
    save_config(config, results_dir / "config.yaml")

    # ------------------------------------------------------------------ #
    #  Ensure task split exists                                          #
    # ------------------------------------------------------------------ #
    logger.info("Ensuring task split exists (seed=%d)...", seed)
    task_split = ensure_split_exists(seed=seed)
    logger.info("Split: %d train, %d holdout",
                len(task_split.train_tasks), len(task_split.holdout_tasks))

    # ------------------------------------------------------------------ #
    #  Initialize metrics recorder                                       #
    # ------------------------------------------------------------------ #
    metrics_cfg = config.get("metrics", {})
    recorder = MetricsRecorder(
        exp_id=exp_id,
        phase=phase,
        results_dir=str(results_dir),
        swanlab_project=metrics_cfg.get("swanlab_project",
                                         "agentic-rl-tau-bench"),
        swanlab_run_name=metrics_cfg.get("swanlab_run_name",
                                          f"{exp_id}_{phase}"),
    )

    # ------------------------------------------------------------------ #
    #  Run evaluation                                                    #
    # ------------------------------------------------------------------ #
    logger.info("Starting tau-bench evaluation...")
    evaluator = TauBenchEvaluator(config=config, results_dir=str(results_dir))

    eval_start = time.time()
    eval_result = evaluator.run()
    eval_elapsed = time.time() - eval_start

    # Capture GPU stats
    gpu_stats = evaluator.get_gpu_stats()
    gpu_stats_path = results_dir / "gpu_stats.json"
    with open(gpu_stats_path, "w", encoding="utf-8") as f:
        json.dump(gpu_stats, f, indent=2)

    if eval_result["returncode"] != 0:
        logger.error("tau-bench evaluation FAILED (exit code %d)",
                     eval_result["returncode"])
        logger.error("See %s for details", evaluator.log_file)
        recorder.record_summary({
            "status": "failed",
            "returncode": eval_result["returncode"],
            "eval_log": str(evaluator.log_file),
        })
        recorder.finish()
        sys.exit(1)

    logger.info("Evaluation completed in %.1f seconds", eval_elapsed)

    # ------------------------------------------------------------------ #
    #  Process trajectories into metrics                                 #
    # ------------------------------------------------------------------ #
    logger.info("Processing trajectories...")
    trajectories = evaluator.load_trajectories()
    logger.info("Loaded %d trajectories", len(trajectories))

    metrics_proc = MetricsProcessor(
        task_split=task_split,
        tokenizer_model=config["agent_model"]["name"],
    )

    per_task = []
    for traj in trajectories:
        per_task.append(metrics_proc.process_trajectory(traj))

    summary = metrics_proc.compute_summary(
        per_task=per_task,
        eval_time_seconds=eval_elapsed,
        gpu_stats=gpu_stats,
    )

    # ------------------------------------------------------------------ #
    #  Failure analysis                                                  #
    # ------------------------------------------------------------------ #
    logger.info("Running failure analysis...")
    max_turns = config["tau_bench"]["max_turns"]
    max_ctx = config["agent_model"]["vllm"]["max_model_len"]
    analyzer = TrajectoryAnalyzer(
        max_turns=max_turns,
        max_context_length=max_ctx,
    )
    failure_analysis = analyzer.analyze_all(trajectories)

    # Update summary with failure mode distribution
    summary["failure_mode_distribution"] = failure_analysis.get(
        "failure_mode_distribution", {}
    )

    # ------------------------------------------------------------------ #
    #  Record all metrics                                                #
    # ------------------------------------------------------------------ #
    logger.info("Recording metrics (JSON + SwanLab)...")
    recorder.record_summary(summary)
    recorder.record_per_task(per_task)
    recorder.write_json(failure_analysis, "failure_analysis.json")

    # ------------------------------------------------------------------ #
    #  Bottleneck analysis                                               #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    #  Write human-readable summary                                      #
    # ------------------------------------------------------------------ #
    summary_md = results_dir / "summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(_format_summary(summary, failure_analysis, eval_elapsed))
    logger.info("Summary written to %s", summary_md)

    recorder.finish()
    logger.info("=== Evaluation complete ===")


def _format_summary(summary: dict, failure_analysis: dict, elapsed: float) -> str:
    """Format a human-readable summary."""
    lines = ["# Evaluation Summary\n"]
    lines.append(f"**Task success rate:** {summary['task_success_rate']:.1%}")
    lines.append(f"**Successful tasks:** {summary['successful_tasks']}/{summary['total_tasks']}")
    lines.append(f"**SFT train group:** {summary['sft_train_group']['success_rate']:.1%} "
                 f"({summary['sft_train_group']['successful']}/{summary['sft_train_group']['total']})")
    lines.append(f"**Holdout group:** {summary['holdout_group']['success_rate']:.1%} "
                 f"({summary['holdout_group']['successful']}/{summary['holdout_group']['total']})")
    lines.append(f"**Eval time:** {elapsed:.1f}s")
    lines.append(f"**Throughput:** {summary['throughput_tasks_per_min']:.1f} tasks/min\n")

    lines.append("## Failure Mode Distribution\n")
    lines.append("| Mode | Count | Percentage |")
    lines.append("|------|-------|------------|")
    for mode, stats in summary.get("failure_mode_distribution", {}).items():
        lines.append(f"| {mode} | {stats['count']} | {stats['pct']:.1f}% |")

    lines.append(f"\n## Trajectory Stats\n")
    lines.append(f"- Avg turns: {summary['avg_trajectory_length_turns']:.1f}")
    lines.append(f"- Max turns: {summary['max_trajectory_length_turns']}")
    lines.append(f"- Avg context: {summary['avg_context_length_tokens']:.0f} tokens")
    lines.append(f"- P95 context: {summary['p95_context_length_tokens']:.0f} tokens")
    lines.append(f"- Total tool calls: {summary['total_tool_calls']}")
    lines.append(f"- Tool call validity: {summary['tool_call_validity_rate']:.1%}")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
