"""Rank bottlenecks by impact and generate optimization direction proposals.

Takes failure analysis + metrics as input, ranks bottlenecks by
frequency x severity, and generates a human-readable report.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Severity weights for impact scoring
SEVERITY_WEIGHTS = {
    "high": 1.0,
    "medium": 0.5,
    "low": 0.2,
}


class BottleneckRanker:
    """Rank bottlenecks and propose optimization directions.

    Args:
        results_dir: Directory containing metrics and analysis outputs.
    """

    def __init__(self, results_dir: str | Path):
        self.results_dir = Path(results_dir)
        self.analysis_dir = self.results_dir.parent / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.analysis_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def rank(
        self,
        failure_analysis: dict,
        summary_metrics: dict,
    ) -> dict:
        """Rank bottlenecks by impact (frequency x severity).

        Args:
            failure_analysis: Output from TrajectoryAnalyzer.analyze_all().
            summary_metrics: Output from MetricsProcessor.compute_summary().

        Returns:
            Structured bottleneck report dict.
        """
        total_failures = failure_analysis.get("total_failures", 0)
        dist = failure_analysis.get("failure_mode_distribution", {})

        # --- Correctness bottlenecks (failure modes) ---
        correctness_bottlenecks = []
        for mode, stats in dist.items():
            count = stats.get("count", 0)
            if count == 0:
                continue
            frequency = count / total_failures if total_failures > 0 else 0.0
            severity = self._severity_for_mode(mode)
            impact = frequency * SEVERITY_WEIGHTS[severity]
            correctness_bottlenecks.append({
                "category": "correctness",
                "mode": mode,
                "description": stats.get("example_task_ids", []),
                "count": count,
                "frequency": round(frequency, 4),
                "severity": severity,
                "impact_score": round(impact, 4),
            })
        correctness_bottlenecks.sort(key=lambda b: b["impact_score"], reverse=True)

        # --- Quality bottleneck (success rate gap) ---
        success_rate = summary_metrics.get("task_success_rate", 0.0)
        quality_bottleneck = {
            "category": "quality",
            "description": f"Overall success rate: {success_rate:.1%}",
            "impact_score": round(1.0 - success_rate, 4),
            "severity": "high" if success_rate < 0.3 else "medium",
        }

        # --- Performance bottleneck ---
        throughput = summary_metrics.get("throughput_tasks_per_min", 0.0)
        latency = summary_metrics.get("avg_latency_per_task_seconds", 0.0)
        perf_bottleneck = {
            "category": "performance",
            "description": f"Throughput: {throughput:.1f} tasks/min, "
                           f"avg latency: {latency:.0f}s/task" if latency else "",
            "impact_score": 0.3,  # placeholder — low priority for research
            "severity": "low",
        }

        # --- Memory bottleneck ---
        gpu_mem = summary_metrics.get("gpu_peak_memory_gb", 0.0)
        mem_bottleneck = {
            "category": "memory",
            "description": f"GPU peak memory: {gpu_mem:.1f} GB / 80 GB per GPU",
            "impact_score": round(gpu_mem / 80.0, 4) if gpu_mem > 0 else 0.0,
            "severity": "high" if gpu_mem > 70 else "medium" if gpu_mem > 50 else "low",
        }

        # --- Reliability bottleneck ---
        reliability_bottleneck = {
            "category": "reliability",
            "description": "Timeout/crash rate (to be measured during eval)",
            "impact_score": 0.0,
            "severity": "low",
        }

        # --- Combine and rank ---
        all_bottlenecks = (
            correctness_bottlenecks
            + [quality_bottleneck, perf_bottleneck, mem_bottleneck,
               reliability_bottleneck]
        )
        all_bottlenecks.sort(key=lambda b: b["impact_score"], reverse=True)

        ranked = []
        for rank, b in enumerate(all_bottlenecks, 1):
            ranked.append({"rank": rank, **b})

        # --- Optimization directions (top <=5) ---
        directions = self._propose_directions(ranked[:5], failure_analysis)

        report = {
            "bottleneck_ranking": ranked,
            "failure_mode_summary": {
                "total_failures": total_failures,
                "distribution": dist,
                "earliest_causal_failure":
                    failure_analysis.get("earliest_causal_failure", {}),
            },
            "optimization_directions": directions,
        }
        return report

    def _severity_for_mode(self, mode: str) -> str:
        """Map failure modes to severity levels."""
        high_impact = {
            "tool_call_error", "context_overflow", "wrong_action",
        }
        medium_impact = {
            "max_turns_exceeded", "premature_termination", "loop",
        }
        if mode in high_impact:
            return "high"
        if mode in medium_impact:
            return "medium"
        return "low"

    def _propose_directions(
        self, top_bottlenecks: list[dict], failure_analysis: dict,
    ) -> list[dict]:
        """Generate optimization direction proposals from top bottlenecks."""
        directions = []
        for b in top_bottlenecks:
            mode = b.get("mode", b.get("category", ""))
            if mode == "tool_call_error":
                directions.append({
                    "direction": "SFT warm-start with teacher trajectories",
                    "hypothesis": "Supervised fine-tuning on successful "
                                  "trajectories from Qwen2.5-72B-AWQ will "
                                  "teach correct tool-call format and reduce "
                                  "JSON parse failures",
                    "expected_impact": "+10-20% success rate",
                    "experiment_cost": "~4 GPU-hours (data gen + SFT)",
                    "risk": "Teacher success rate may be too low; "
                            "SFT may overfit to teacher patterns",
                    "confidence": "high",
                })
            elif mode == "wrong_action":
                directions.append({
                    "direction": "GRPO with trajectory-level reward",
                    "hypothesis": "RL with binary task-success reward will "
                                  "optimize for correct action sequences, "
                                  "not just valid tool calls",
                    "expected_impact": "+5-15% success rate over SFT",
                    "experiment_cost": "~8 GPU-hours per GRPO run",
                    "risk": "Sparse reward may cause high variance; "
                            "low rollout success rate limits learning signal",
                    "confidence": "medium",
                })
            elif mode == "context_overflow":
                directions.append({
                    "direction": "Context compression / observation truncation",
                    "hypothesis": "Truncating tool outputs and compressing "
                                  "observation tokens will prevent context "
                                  "overflow without losing key information",
                    "expected_impact": "Reduce context overflow failures to ~0",
                    "experiment_cost": "~2 GPU-hours to implement + test",
                    "risk": "Compression may remove information needed for "
                            "correct task completion",
                    "confidence": "medium",
                })
            elif mode == "loop":
                directions.append({
                    "direction": "Anti-loop penalty in reward shaping",
                    "hypothesis": "Adding a penalty for repeated actions in "
                                  "the reward function will discourage loops",
                    "expected_impact": "Eliminate loop failures",
                    "experiment_cost": "~1 GPU-hour to implement + test",
                    "risk": "Penalty may discourage legitimate retries",
                    "confidence": "high",
                })
            elif b.get("category") == "quality":
                directions.append({
                    "direction": "Multi-stage training: SFT then GRPO",
                    "hypothesis": "SFT warm-start followed by GRPO will "
                                  "combine imitation learning benefits with "
                                  "RL optimization",
                    "expected_impact": "+15-25% over zero-shot baseline",
                    "experiment_cost": "~12 GPU-hours total",
                    "risk": "GRPO may degrade SFT capabilities if "
                            "hyperparameters are poorly tuned",
                    "confidence": "medium",
                })

        # Ensure at least one direction
        if not directions:
            directions.append({
                "direction": "SFT warm-start (default next step)",
                "hypothesis": "Supervised fine-tuning on successful teacher "
                              "trajectories is the natural first step after "
                              "establishing the zero-shot baseline",
                "expected_impact": "+10-20% success rate",
                "experiment_cost": "~4 GPU-hours",
                "risk": "Teacher success rate may be insufficient",
                "confidence": "high",
            })

        return directions[:5]

    def save_report(self, report: dict) -> tuple[Path, Path]:
        """Save the bottleneck report as JSON and Markdown.

        Returns:
            Tuple of (json_path, md_path).
        """
        json_path = self.metrics_dir / "bottleneck_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        md_path = self.analysis_dir / "bottleneck_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._format_markdown(report))

        logger.info("Bottleneck report saved: %s, %s", json_path, md_path)
        return json_path, md_path

    @staticmethod
    def _format_markdown(report: dict) -> str:
        """Format the bottleneck report as human-readable Markdown."""
        lines = ["# Bottleneck Analysis Report\n"]

        lines.append("## Bottleneck Ranking\n")
        lines.append("| Rank | Category | Description | Impact | Severity |")
        lines.append("|------|----------|-------------|--------|----------|")
        for b in report.get("bottleneck_ranking", []):
            desc = b.get("description", b.get("mode", ""))
            if isinstance(desc, list):
                desc = ", ".join(str(d) for d in desc[:3])
            lines.append(
                f"| {b['rank']} | {b['category']} | {desc} | "
                f"{b['impact_score']:.4f} | {b['severity']} |"
            )

        lines.append("\n## Failure Mode Distribution\n")
        summary = report.get("failure_mode_summary", {})
        dist = summary.get("distribution", {})
        lines.append("| Mode | Count | Percentage | Example Tasks |")
        lines.append("|------|-------|------------|---------------|")
        for mode, stats in dist.items():
            examples = ", ".join(str(t) for t in stats.get("example_task_ids", []))
            lines.append(
                f"| {mode} | {stats['count']} | {stats['pct']:.1f}% | {examples} |"
            )

        lines.append("\n## Optimization Directions\n")
        for i, d in enumerate(report.get("optimization_directions", []), 1):
            lines.append(f"### {i}. {d['direction']}\n")
            lines.append(f"- **Hypothesis:** {d['hypothesis']}")
            lines.append(f"- **Expected Impact:** {d['expected_impact']}")
            lines.append(f"- **Experiment Cost:** {d['experiment_cost']}")
            lines.append(f"- **Risk:** {d['risk']}")
            lines.append(f"- **Confidence:** {d['confidence']}\n")

        return "\n".join(lines)
