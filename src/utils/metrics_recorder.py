"""Unified metrics recorder: SwanLab + local JSON dual-channel.

Every stage uses this recorder. JSON is the authoritative record (never lost).
SwanLab provides visualization and cross-experiment comparison.
If SwanLab cloud is unreachable, auto-fallback to local-only mode.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetricsRecorder:
    """Dual-channel metrics recorder (JSON + SwanLab).

    Usage:
        recorder = MetricsRecorder(
            exp_id="exp_001",
            phase="baseline_eval",
            results_dir="results/baseline",
            swanlab_project="agentic-rl-tau-bench",
            swanlab_run_name="exp_001_baseline",
        )
        recorder.record_summary({"task_success_rate": 0.12})
        recorder.record_step_metrics({"step": 1, "loss": 2.3})
        recorder.record_per_task([{"task_id": "task_0", "success": True}])
        recorder.finish()
    """

    def __init__(
        self,
        exp_id: str,
        phase: str,
        results_dir: str | Path,
        swanlab_project: str = "agentic-rl-tau-bench",
        swanlab_run_name: str | None = None,
    ):
        self.exp_id = exp_id
        self.phase = phase
        self.results_dir = Path(results_dir)
        self.metrics_dir = self.results_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.swanlab_project = swanlab_project
        self.swanlab_run_name = swanlab_run_name or f"{exp_id}_{phase}"
        self.swanlab_offline = False
        self._swanlab_run = None
        self._step_buffer: list[dict] = []

        self._init_swanlab()

    def _init_swanlab(self) -> None:
        """Initialize SwanLab. Fallback to offline if unavailable."""
        if os.environ.get("SWANLAB_MODE", "").lower() == "disabled":
            self.swanlab_offline = True
            logger.info("SwanLab disabled via SWANLAB_MODE=disabled")
            return

        try:
            import swanlab

            self._swanlab_run = swanlab.init(
                project=self.swanlab_project,
                experiment_name=self.swanlab_run_name,
                description=f"{self.exp_id} - {self.phase}",
                config={"exp_id": self.exp_id, "phase": self.phase},
            )
            logger.info(
                "SwanLab initialized: project=%s, run=%s",
                self.swanlab_project,
                self.swanlab_run_name,
            )
        except Exception as e:
            self.swanlab_offline = True
            logger.warning(
                "SwanLab unavailable (%s). Falling back to local JSON only.",
                e,
            )

    def _write_json(self, data: Any, filename: str) -> Path:
        """Write data to a JSON file in the metrics directory."""
        filepath = self.metrics_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return filepath

    def _append_jsonl(self, data: dict, filename: str) -> Path:
        """Append a record to a JSONL file."""
        filepath = self.metrics_dir / filename
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
        return filepath

    def _log_swanlab(self, metrics: dict, step: int | None = None) -> None:
        """Log metrics to SwanLab. Silently skip if offline."""
        if self.swanlab_offline or self._swanlab_run is None:
            return
        try:
            import swanlab

            swanlab.log(metrics, step=step)
        except Exception as e:
            logger.warning("SwanLab log failed (%s). Continuing with JSON only.", e)
            self.swanlab_offline = True

    def record_summary(self, metrics: dict, filename: str | None = None) -> Path:
        """Record summary metrics for the current phase.

        Args:
            metrics: Dict of metric name -> value.
            filename: Override JSON filename. Default: {phase}_metrics.json

        Returns:
            Path to the written JSON file.
        """
        fname = filename or f"{self.phase}_metrics.json"
        # Add metadata
        record = {
            "exp_id": self.exp_id,
            "phase": self.phase,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "swanlab_offline": self.swanlab_offline,
            **metrics,
        }
        filepath = self._write_json(record, fname)
        self._log_swanlab(metrics)
        logger.info("Summary metrics written to %s", filepath)
        return filepath

    def record_step_metrics(self, metrics: dict, filename: str | None = None) -> Path:
        """Record per-step metrics (e.g., training loss per step).

        Appends to JSONL for streaming and logs to SwanLab.

        Args:
            metrics: Dict including 'step' key and metric values.
            filename: Override JSONL filename. Default: {phase}_step_metrics.jsonl

        Returns:
            Path to the appended JSONL file.
        """
        fname = filename or f"{self.phase}_step_metrics.jsonl"
        record = {
            "exp_id": self.exp_id,
            "phase": self.phase,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        filepath = self._append_jsonl(record, fname)

        step = metrics.get("step")
        # Filter out non-numeric values for SwanLab
        swanlab_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and k != "step"
        }
        self._log_swanlab(swanlab_metrics, step=step)
        return filepath

    def record_per_task(self, results: list[dict], filename: str | None = None) -> Path:
        """Record per-task detailed results.

        Args:
            results: List of per-task result dicts.
            filename: Override JSON filename. Default: per_task_results.json

        Returns:
            Path to the written JSON file.
        """
        fname = filename or "per_task_results.json"
        record = {
            "exp_id": self.exp_id,
            "phase": self.phase,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_tasks": len(results),
            "results": results,
        }
        return self._write_json(record, fname)

    def record_jsonl(self, data: dict, filename: str) -> Path:
        """Append arbitrary data to a named JSONL file."""
        return self._append_jsonl(data, filename)

    def write_json(self, data: Any, filename: str) -> Path:
        """Write arbitrary data to a named JSON file in metrics dir."""
        return self._write_json(data, filename)

    def finish(self) -> None:
        """Finalize recording. Close SwanLab run if active."""
        if not self.swanlab_offline and self._swanlab_run is not None:
            try:
                import swanlab

                swanlab.finish()
                logger.info("SwanLab run finished.")
            except Exception as e:
                logger.warning("SwanLab finish failed (%s).", e)
