"""Classify trajectory failures and identify earliest causal failure point.

Per AGENTS.md Section 9: when a trajectory fails, inspect the full failure
chain and identify the *earliest* causal failure, not just the final error.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Failure mode labels
TOOL_CALL_ERROR = "tool_call_error"
CONTEXT_OVERFLOW = "context_overflow"
MAX_TURNS_EXCEEDED = "max_turns_exceeded"
WRONG_ACTION = "wrong_action"
USER_SIMULATOR_BREAKDOWN = "user_simulator_breakdown"
PREMATURE_TERMINATION = "premature_termination"
LOOP = "loop"
OTHER = "other"

ALL_MODES = [
    TOOL_CALL_ERROR,
    CONTEXT_OVERFLOW,
    MAX_TURNS_EXCEEDED,
    WRONG_ACTION,
    USER_SIMULATOR_BREAKDOWN,
    PREMATURE_TERMINATION,
    LOOP,
    OTHER,
]

# Patterns for detecting errors in message content
_JSON_ERROR_RE = re.compile(r"json.*(parse|decode|invalid|error)", re.IGNORECASE)
_INVALID_TOOL_RE = re.compile(r"(invalid|unknown|unrecognized).*(tool|function)", re.IGNORECASE)
_ARGUMENT_ERROR_RE = re.compile(r"(invalid|missing|required).*(argument|param)", re.IGNORECASE)
_MAX_TURN_RE = re.compile(r"(max|maximum).*(turn|round|step)", re.IGNORECASE)
_CONTEXT_LEN_RE = re.compile(r"(context|token).*(length|limit|exceed|overflow|too long)", re.IGNORECASE)
_SIMULATOR_BREAK_RE = re.compile(r"(simulator|user).*(fail|break|error|abandon|give up)", re.IGNORECASE)


class TrajectoryAnalyzer:
    """Analyze failed trajectories and classify failure modes.

    Args:
        max_turns: Maximum turns allowed (for max_turns_exceeded detection).
        max_context_length: Maximum context length in tokens.
    """

    def __init__(self, max_turns: int = 30, max_context_length: int = 32768):
        self.max_turns = max_turns
        self.max_context_length = max_context_length

    def analyze_trajectory(self, traj: dict) -> dict:
        """Classify a single trajectory failure mode and earliest causal
        failure point.

        Returns:
            Dict with keys: task_id, success, failure_mode,
            earliest_causal_failure_turn, earliest_causal_failure_description,
            evidence.
        """
        task_id = str(traj.get("task_id", traj.get("id", "unknown")))
        reward = traj.get("reward", 0)
        success = bool(reward) if isinstance(reward, (int, float)) else False

        if success:
            return {
                "task_id": task_id,
                "success": True,
                "failure_mode": "success",
                "earliest_causal_failure_turn": None,
                "earliest_causal_failure_description": "",
                "evidence": {},
            }

        messages = self._extract_messages(traj)
        n_turns = len(messages)

        signals = []
        for i, msg in enumerate(messages):
            content = self._get_text(msg)

            if _JSON_ERROR_RE.search(content):
                signals.append({"turn": i, "mode": TOOL_CALL_ERROR,
                                "detail": "JSON parse error detected"})
            if _INVALID_TOOL_RE.search(content):
                signals.append({"turn": i, "mode": TOOL_CALL_ERROR,
                                "detail": "Invalid tool name detected"})
            if _ARGUMENT_ERROR_RE.search(content):
                signals.append({"turn": i, "mode": TOOL_CALL_ERROR,
                                "detail": "Invalid argument detected"})
            if _MAX_TURN_RE.search(content):
                signals.append({"turn": i, "mode": MAX_TURNS_EXCEEDED,
                                "detail": "Max turns reached"})
            if _CONTEXT_LEN_RE.search(content):
                signals.append({"turn": i, "mode": CONTEXT_OVERFLOW,
                                "detail": "Context length exceeded"})
            if _SIMULATOR_BREAK_RE.search(content):
                signals.append({"turn": i, "mode": USER_SIMULATOR_BREAKDOWN,
                                "detail": "Simulator breakdown detected"})

        loop_turn = self._detect_loop(messages)
        if loop_turn is not None:
            signals.append({"turn": loop_turn, "mode": LOOP,
                            "detail": "Repeated identical actions without progress"})

        premature_turn = self._detect_premature_termination(messages)
        if premature_turn is not None:
            signals.append({"turn": premature_turn, "mode": PREMATURE_TERMINATION,
                            "detail": "Agent declared completion prematurely"})

        if n_turns >= self.max_turns:
            signals.append({"turn": n_turns - 1, "mode": MAX_TURNS_EXCEEDED,
                            "detail": f"Reached max turns ({self.max_turns})"})

        if signals:
            signals.sort(key=lambda s: s["turn"])
            primary = signals[0]
            failure_mode = primary["mode"]
            earliest_turn = primary["turn"]
            description = primary["detail"]
        else:
            failure_mode = WRONG_ACTION
            earliest_turn = n_turns - 1 if n_turns > 0 else None
            description = ("No explicit error detected; agent executed valid "
                           "actions but final database state did not match")

        return {
            "task_id": task_id,
            "success": False,
            "failure_mode": failure_mode,
            "earliest_causal_failure_turn": earliest_turn,
            "earliest_causal_failure_description": description,
            "evidence": {"signals": signals},
            "n_turns": n_turns,
        }

    def analyze_all(self, trajectories: list[dict]) -> dict:
        """Analyze all trajectories and produce a failure analysis report."""
        per_traj = [self.analyze_trajectory(t) for t in trajectories]
        failed = [t for t in per_traj if not t["success"]]

        distribution: dict[str, dict] = {}
        for mode in ALL_MODES:
            mode_results = [t for t in failed if t["failure_mode"] == mode]
            distribution[mode] = {
                "count": len(mode_results),
                "pct": len(mode_results) / len(failed) * 100 if failed else 0.0,
                "example_task_ids": [t["task_id"] for t in mode_results[:5]],
            }

        earliest_stats: dict[str, dict] = {}
        for mode in ALL_MODES:
            mode_results = [t for t in failed if t["failure_mode"] == mode]
            if mode_results:
                turns = [t["earliest_causal_failure_turn"]
                         for t in mode_results
                         if t["earliest_causal_failure_turn"] is not None]
                earliest_stats[mode] = {
                    "avg_turn": sum(turns) / len(turns) if turns else None,
                    "common_patterns": list(set(
                        t["earliest_causal_failure_description"]
                        for t in mode_results))[:3],
                }

        return {
            "total_trajectories": len(per_traj),
            "total_failures": len(failed),
            "total_successes": len(per_traj) - len(failed),
            "failure_mode_distribution": distribution,
            "earliest_causal_failure": earliest_stats,
            "per_trajectory": per_traj,
        }

    def _extract_messages(self, traj: dict) -> list[dict]:
        for key in ("messages", "trajectory", "chat_history", "traj"):
            if key in traj and isinstance(traj[key], list):
                return traj[key]
        info = traj.get("info", {})
        if isinstance(info, dict):
            for key in ("messages", "trajectory", "chat_history"):
                if key in info and isinstance(info[key], list):
                    return info[key]
        return []

    @staticmethod
    def _get_text(msg: dict) -> str:
        content = msg.get("content", "")
        if isinstance(content, list):
            return " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )
        return str(content)

    def _detect_loop(self, messages: list[dict]) -> int | None:
        """Detect repeated identical assistant messages (3+ in a row)."""
        assistant_texts = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                text = self._get_text(msg)
                normalized = " ".join(text.split())[:200]
                assistant_texts.append((i, normalized))

        for i in range(2, len(assistant_texts)):
            if (assistant_texts[i][1] == assistant_texts[i-1][1]
                    and assistant_texts[i][1] == assistant_texts[i-2][1]
                    and assistant_texts[i][1]):
                return assistant_texts[i-2][0]
        return None

    def _detect_premature_termination(self, messages: list[dict]) -> int | None:
        """Detect agent declaring completion well before the final turn."""
        completion_re = re.compile(
            r"\b(done|complete|finished|task complete|all set)\b", re.IGNORECASE)
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                text = self._get_text(msg)
                if completion_re.search(text) and i < len(messages) - 2:
                    return i
        return None
