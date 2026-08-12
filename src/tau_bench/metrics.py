"""Post-process raw tau-bench trajectories into structured metrics.

Token counting uses the Qwen2.5 tokenizer for accuracy (never character
counts). All metrics are reported with sft_train(40)/holdout(10) breakdown.
"""

import json
import logging
import math
import statistics
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetricsProcessor:
    """Convert raw trajectory logs into the structured metrics defined in
    the Stage Metrics Spec.

    Args:
        task_split: TaskSplit instance for train/holdout grouping.
        tokenizer_model: HuggingFace model name for the tokenizer.
    """

    def __init__(self, task_split, tokenizer_model: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.task_split = task_split
        self.tokenizer_model = tokenizer_model
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazily load the Qwen2.5 tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_model, trust_remote_code=True,
            )
            logger.info("Loaded tokenizer: %s", self.tokenizer_model)
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens using the Qwen2.5 tokenizer."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    # ------------------------------------------------------------------ #
    #  Per-trajectory processing                                         #
    # ------------------------------------------------------------------ #

    def process_trajectory(self, traj: dict) -> dict:
        """Extract metrics from a single trajectory log.

        Handles various tau-bench trajectory formats gracefully.
        """
        task_id = str(traj.get("task_id", traj.get("id", "unknown")))
        reward = traj.get("reward", 0)
        success = bool(reward) if isinstance(reward, (int, float)) else False
        group = self.task_split.get_group(task_id)

        # Extract message list — tau-bench stores it under various keys
        messages = self._extract_messages(traj)
        n_turns = len(messages)

        # Token counting per message
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        tool_calls = 0
        json_parse_failures = 0
        invalid_tool_names = 0
        invalid_arguments = 0
        tool_execution_failures = 0
        valid_tool_calls = 0

        cumulative_context = []
        context_lengths = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if isinstance(content, list):
                # content can be a list of parts (e.g. tool calls)
                content_text = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            else:
                content_text = str(content)

            n_tok = self.count_tokens(content_text)
            total_tokens += n_tok

            if role in ("user", "tool", "system"):
                total_input_tokens += n_tok
            elif role == "assistant":
                total_output_tokens += n_tok

            # Tool call analysis
            tool_call_list = msg.get("tool_calls", [])
            if tool_call_list:
                for tc in tool_call_list:
                    tool_calls += 1
                    if self._is_valid_tool_call(tc):
                        valid_tool_calls += 1
                    else:
                        if not self._has_valid_json(tc):
                            json_parse_failures += 1
                        if not self._has_valid_tool_name(tc):
                            invalid_tool_names += 1
                        if not self._has_valid_arguments(tc):
                            invalid_arguments += 1

            # Track cumulative context growth
            cumulative_context.append(content_text)
            context_lengths.append(
                self.count_tokens("\n".join(cumulative_context))
            )

        tool_call_validity_rate = (
            valid_tool_calls / tool_calls if tool_calls > 0 else 1.0
        )

        return {
            "task_id": task_id,
            "success": success,
            "reward": reward,
            "group": group,
            "trajectory_length_turns": n_turns,
            "total_tokens": total_tokens,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "context_length_tokens": context_lengths[-1] if context_lengths else 0,
            "max_context_per_turn": max(context_lengths) if context_lengths else 0,
            "context_lengths_per_turn": context_lengths,
            "total_tool_calls": tool_calls,
            "valid_tool_calls": valid_tool_calls,
            "tool_call_validity_rate": tool_call_validity_rate,
            "json_parse_failures": json_parse_failures,
            "invalid_tool_names": invalid_tool_names,
            "invalid_arguments": invalid_arguments,
            "tool_execution_failures": tool_execution_failures,
        }

    def _extract_messages(self, traj: dict) -> list[dict]:
        """Extract the message list from a trajectory dict.

        tau-bench stores messages under various keys depending on version.
        """
        for key in ("messages", "trajectory", "chat_history", "traj"):
            if key in traj and isinstance(traj[key], list):
                return traj[key]
        # Some versions nest under "info"
        info = traj.get("info", {})
        if isinstance(info, dict):
            for key in ("messages", "trajectory", "chat_history"):
                if key in info and isinstance(info[key], list):
                    return info[key]
        logger.warning("Could not extract messages from trajectory keys: %s",
                        list(traj.keys()))
        return []

    def _is_valid_tool_call(self, tc: dict) -> bool:
        """Check if a tool call is structurally valid."""
        return (
            self._has_valid_json(tc)
            and self._has_valid_tool_name(tc)
            and self._has_valid_arguments(tc)
        )

    def _has_valid_json(self, tc: dict) -> bool:
        fn = tc.get("function", tc)
        args = fn.get("arguments", "")
        if isinstance(args, str):
            try:
                json.loads(args)
                return True
            except (json.JSONDecodeError, TypeError):
                return False
        if isinstance(args, dict):
            return True
        return False

    def _has_valid_tool_name(self, tc: dict) -> bool:
        fn = tc.get("function", tc)
        name = fn.get("name", "")
        return bool(name) and isinstance(name, str)

    def _has_valid_arguments(self, tc: dict) -> bool:
        fn = tc.get("function", tc)
        args = fn.get("arguments", "")
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                return isinstance(parsed, dict)
            except (json.JSONDecodeError, TypeError):
                return False
        return isinstance(args, dict)

    # ------------------------------------------------------------------ #
    #  Aggregation                                                       #
    # ------------------------------------------------------------------ #

    def compute_summary(
        self,
        per_task: list[dict],
        eval_time_seconds: float = 0.0,
        gpu_stats: dict | None = None,
    ) -> dict:
        """Aggregate per-task results into summary metrics.

        Args:
            per_task: List of per-trajectory metric dicts.
            eval_time_seconds: Total evaluation wall-clock time.
            gpu_stats: GPU memory/utilization snapshot.

        Returns:
            Summary dict matching the Stage Metrics Spec.
        """
        total = len(per_task)
        successful = sum(1 for t in per_task if t["success"])
        success_rate = successful / total if total > 0 else 0.0

        # Group breakdowns
        train_results = [t for t in per_task if t["group"] == "sft_train"]
        holdout_results = [t for t in per_task if t["group"] == "holdout"]

        train_success = sum(1 for t in train_results if t["success"])
        holdout_success = sum(1 for t in holdout_results if t["success"])

        zero_success = [t["task_id"] for t in per_task if not t["success"]]

        # Trajectory-level stats
        traj_lengths = [t["trajectory_length_turns"] for t in per_task]
        context_lengths = [t["context_length_tokens"] for t in per_task]
        input_tokens = [t["total_input_tokens"] for t in per_task]
        output_tokens = [t["total_output_tokens"] for t in per_task]

        # Tool-use stats
        total_tool_calls = sum(t["total_tool_calls"] for t in per_task)
        total_valid = sum(t["valid_tool_calls"] for t in per_task)
        total_json_fail = sum(t["json_parse_failures"] for t in per_task)
        total_invalid_names = sum(t["invalid_tool_names"] for t in per_task)
        total_invalid_args = sum(t["invalid_arguments"] for t in per_task)

        tool_call_counts = [t["total_tool_calls"] for t in per_task]

        summary = {
            # Task-level
            "total_tasks": total,
            "successful_tasks": successful,
            "task_success_rate": success_rate,
            "sft_train_group": {
                "total": len(train_results),
                "successful": train_success,
                "success_rate": (
                    train_success / len(train_results)
                    if train_results else 0.0
                ),
            },
            "holdout_group": {
                "total": len(holdout_results),
                "successful": holdout_success,
                "success_rate": (
                    holdout_success / len(holdout_results)
                    if holdout_results else 0.0
                ),
            },
            "zero_success_tasks": zero_success,
            # Trajectory-level
            "avg_trajectory_length_turns": (
                statistics.mean(traj_lengths) if traj_lengths else 0.0
            ),
            "max_trajectory_length_turns": max(traj_lengths) if traj_lengths else 0,
            "avg_context_length_tokens": (
                statistics.mean(context_lengths) if context_lengths else 0.0
            ),
            "max_context_length_tokens": max(context_lengths) if context_lengths else 0,
            "p95_context_length_tokens": self._percentile(context_lengths, 95),
            "p99_context_length_tokens": self._percentile(context_lengths, 99),
            "avg_input_tokens_per_turn": (
                statistics.mean(input_tokens) if input_tokens else 0.0
            ),
            "avg_output_tokens_per_turn": (
                statistics.mean(output_tokens) if output_tokens else 0.0
            ),
            "total_input_tokens": sum(input_tokens),
            "total_output_tokens": sum(output_tokens),
            # Tool-use
            "total_tool_calls": total_tool_calls,
            "tool_call_validity_rate": (
                total_valid / total_tool_calls if total_tool_calls > 0 else 1.0
            ),
            "json_parse_failures": total_json_fail,
            "invalid_tool_names": total_invalid_names,
            "invalid_arguments": total_invalid_args,
            "tool_execution_failures": 0,  # populated by analyzer
            "tool_call_count_distribution": {
                "avg": statistics.mean(tool_call_counts) if tool_call_counts else 0.0,
                "min": min(tool_call_counts) if tool_call_counts else 0,
                "max": max(tool_call_counts) if tool_call_counts else 0,
            },
            # Infrastructure
            "total_eval_seconds": eval_time_seconds,
            "throughput_tasks_per_min": (
                total / (eval_time_seconds / 60) if eval_time_seconds > 0 else 0.0
            ),
            "avg_latency_per_task_seconds": (
                eval_time_seconds / total if total > 0 else 0.0
            ),
            "gpu_peak_memory_gb": (
                max(
                    (g["memory_used_mb"] for g in gpu_stats.get("gpus", [])),
                    default=0,
                ) / 1024
                if gpu_stats else 0.0
            ),
            "gpu_utilization_pct": (
                statistics.mean(
                    [g["utilization_pct"] for g in gpu_stats.get("gpus", [])]
                )
                if gpu_stats and gpu_stats.get("gpus") else 0.0
            ),
            # Failure modes — populated by TrajectoryAnalyzer
            "failure_mode_distribution": {},
        }
        return summary

    @staticmethod
    def _percentile(data: list, p: float) -> float:
        """Compute the p-th percentile of a list."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(sorted_data[int(k)])
        return float(
            sorted_data[f] * (c - k) + sorted_data[c] * (k - f)
        )
