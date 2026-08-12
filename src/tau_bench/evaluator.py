"""tau-bench evaluator: run evaluations and collect raw trajectories.

Wraps the tau-bench CLI, configuring separate vLLM endpoints for the agent
(port 8000) and the user simulator (port 8001). Trajectories are saved to
a log directory for downstream metric processing.
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default health-check settings
HEALTH_CHECK_TIMEOUT = 300  # seconds
HEALTH_CHECK_INTERVAL = 5  # seconds


class TauBenchEvaluator:
    """Run tau-bench evaluations with dual vLLM backends.

    Args:
        config: Parsed config dict (see configs/baseline_eval.yaml).
        results_dir: Directory to save results and trajectories.
    """

    def __init__(self, config: dict, results_dir: str | Path):
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.traj_dir = self.results_dir / "trajectories"
        self.traj_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.results_dir / "eval.log"

        self.agent_model = config["agent_model"]["name"]
        self.simulator_model = config["simulator_model"]["name"]
        self.agent_port = config["agent_model"]["vllm"]["port"]
        self.simulator_port = config["simulator_model"]["vllm"]["port"]
        self.agent_api_base = f"http://localhost:{self.agent_port}/v1"
        self.simulator_api_base = f"http://localhost:{self.simulator_port}/v1"

        tb = config["tau_bench"]
        self.domain = tb["domain"]
        self.max_turns = tb["max_turns"]
        self.num_rollouts = tb["num_rollouts"]
        self.agent_temperature = tb["agent_temperature"]
        self.simulator_temperature = tb["simulator_temperature"]

    # ------------------------------------------------------------------ #
    #  Health checks                                                     #
    # ------------------------------------------------------------------ #

    def wait_for_services(self, timeout: int = HEALTH_CHECK_TIMEOUT) -> None:
        """Block until both vLLM endpoints respond to /v1/models."""
        import urllib.request
        import urllib.error

        endpoints = [
            ("agent", f"http://localhost:{self.agent_port}/v1/models"),
            ("simulator", f"http://localhost:{self.simulator_port}/v1/models"),
        ]
        deadline = time.time() + timeout
        for name, url in endpoints:
            logger.info("Waiting for %s vLLM at %s ...", name, url)
            while time.time() < deadline:
                try:
                    req = urllib.request.Request(url)
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        if resp.status == 200:
                            logger.info("%s vLLM is ready.", name)
                            break
                except (urllib.error.URLError, OSError):
                    pass
                time.sleep(HEALTH_CHECK_INTERVAL)
            else:
                raise TimeoutError(
                    f"{name} vLLM at {url} not ready within {timeout}s"
                )

    # ------------------------------------------------------------------ #
    #  Run evaluation                                                    #
    # ------------------------------------------------------------------ #

    def build_command(self) -> list[str]:
        """Build the tau-bench CLI command.

        Uses --simulator-api-base to point the user simulator at a separate
        vLLM instance. If the installed tau-bench version does not support
        this flag, set SIMULATOR_API_BASE env var as fallback or patch
        tau-bench source.  [ASSUMPTION — see research/DECISIONS.md D008]
        """
        cmd = [
            "python", "-m", "tau_bench",
            "--mode", "eval",
            "--env", self.domain,
            "--model", self.agent_model,
            "--simulator-model", self.simulator_model,
            "--num-epochs", str(self.num_rollouts),
            "--max-turns", str(self.max_turns),
            "--temperature", str(self.agent_temperature),
            "--simulator-temperature", str(self.simulator_temperature),
            "--api-base", self.agent_api_base,
            "--simulator-api-base", self.simulator_api_base,
            "--log-dir", str(self.traj_dir),
        ]
        return cmd

    def run(self) -> dict[str, Any]:
        """Execute the tau-bench evaluation.

        Returns:
            Dict with keys:
                - returncode: subprocess return code
                - stdout: captured stdout
                - stderr: captured stderr
                - raw_results: parsed JSON results (if available)
                - trajectories: list of trajectory log paths
        """
        self.wait_for_services()

        cmd = self.build_command()
        logger.info("Running tau-bench: %s", " ".join(cmd))

        env = os.environ.copy()
        env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY", "dummy-key")

        with open(self.log_file, "w", encoding="utf-8") as logf:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                timeout=7200,  # 2 hour hard limit
            )
            logf.write(proc.stdout)

        result: dict[str, Any] = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": "",
            "raw_results": None,
            "trajectories": [],
        }

        if proc.returncode != 0:
            logger.error(
                "tau-bench exited with code %d. See %s", proc.returncode, self.log_file
            )
            return result

        # Parse JSON results from stdout (tau-bench prints JSON summary)
        result["raw_results"] = self._parse_stdout_json(proc.stdout)

        # Collect trajectory log paths
        traj_files = sorted(self.traj_dir.glob("*.json"))
        result["trajectories"] = [str(p) for p in traj_files]
        logger.info("Collected %d trajectory files", len(traj_files))

        return result

    # ------------------------------------------------------------------ #
    #  Output parsing                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_stdout_json(stdout: str) -> list[dict] | None:
        """Extract JSON result objects from tau-bench stdout.

        tau-bench prints per-task results as JSON lines and a summary.
        We try to parse each line; non-JSON lines are skipped.
        """
        results: list[dict] = []
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    results.append(obj)
            except json.JSONDecodeError:
                continue
        return results if results else None

    # ------------------------------------------------------------------ #
    #  Trajectory loading                                                #
    # ------------------------------------------------------------------ #

    def load_trajectories(self) -> list[dict]:
        """Load all trajectory JSON files from the log directory."""
        traj_files = sorted(self.traj_dir.glob("*.json"))
        trajectories = []
        for f in traj_files:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    trajectories.append(json.load(fh))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load %s: %s", f, e)
        logger.info("Loaded %d trajectories", len(trajectories))
        return trajectories

    def get_gpu_stats(self) -> dict:
        """Capture GPU memory and utilization snapshot via nvidia-smi."""
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,"
                 "memory.free,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0:
                gpus = []
                for line in proc.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        gpus.append({
                            "index": int(parts[0]),
                            "memory_total_mb": int(parts[1]),
                            "memory_used_mb": int(parts[2]),
                            "memory_free_mb": int(parts[3]),
                            "utilization_pct": int(parts[4]),
                        })
                return {"gpus": gpus, "timestamp": time.time()}
        except (OSError, subprocess.TimeoutExpired):
            pass
        return {"gpus": [], "timestamp": time.time()}
