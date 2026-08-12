"""Fixed task split for tau-bench-airline: 40 train / 10 holdout (seed=42).

The split is deterministic and committed to the repo so every experiment
uses the same task partition. Holdout task IDs are never used for training.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SPLIT_DIR = Path(__file__).resolve().parents[2] / "data" / "task_splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


class TaskSplit:
    """Manage the 40/10 train/holdout split of 50 airline tasks.

    On first use, call ``create_split()`` to generate the JSON files.
    Subsequent calls load from disk.
    """

    def __init__(self, split_dir: Path | str | None = None):
        self.split_dir = Path(split_dir) if split_dir else SPLIT_DIR
        self._train_tasks: list[str] | None = None
        self._holdout_tasks: list[str] | None = None
        self._all_tasks: list[str] | None = None

    # ------------------------------------------------------------------ #
    #  Loading from tau-bench                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_task_ids_from_tau_bench() -> list[str]:
        """Import airline task IDs from the installed tau-bench package.

        Raises:
            ImportError: if tau-bench is not installed.
        """
        from tau_bench.envs.airline.tasks import TASKS  # type: ignore

        ids = [str(t.task_id) if hasattr(t, "task_id") else str(i)
               for i, t in enumerate(TASKS)]
        logger.info("Loaded %d airline task IDs from tau-bench", len(ids))
        return ids

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    def create_split(self, seed: int = 42, n_holdout: int = 10) -> None:
        """Generate and persist the train/holdout split.

        Args:
            seed: Random seed for reproducibility.
            n_holdout: Number of holdout tasks (default 10).
        """
        import random

        all_ids = self._load_task_ids_from_tau_bench()
        if len(all_ids) != 50:
            logger.warning(
                "Expected 50 airline tasks, got %d. Split will proceed "
 "with available tasks.", len(all_ids),
            )

        rng = random.Random(seed)
        shuffled = sorted(all_ids)  # sort for determinism before shuffle
        rng.shuffle(shuffled)

        holdout = sorted(shuffled[:n_holdout])
        train = sorted(shuffled[n_holdout:])

        self._all_tasks = all_ids
        self._train_tasks = train
        self._holdout_tasks = holdout

        self._write_json("all_tasks.json", all_ids)
        self._write_json("sft_train_tasks.json", train)
        self._write_json("holdout_tasks.json", holdout)

        logger.info(
            "Task split created: %d train, %d holdout (seed=%d)",
            len(train), len(holdout), seed,
        )
        logger.info("Holdout task IDs: %s", holdout)

    def load_split(self) -> tuple[list[str], list[str], list[str]]:
        """Load the split from persisted JSON files.

        Returns:
            Tuple of (train_tasks, holdout_tasks, all_tasks).
        """
        self._train_tasks = self._read_json("sft_train_tasks.json")
        self._holdout_tasks = self._read_json("holdout_tasks.json")
        self._all_tasks = self._read_json("all_tasks.json")
        return self._train_tasks, self._holdout_tasks, self._all_tasks

    @property
    def train_tasks(self) -> list[str]:
        if self._train_tasks is None:
            self.load_split()
        return self._train_tasks  # type: ignore[return-value]

    @property
    def holdout_tasks(self) -> list[str]:
        if self._holdout_tasks is None:
            self.load_split()
        return self._holdout_tasks  # type: ignore[return-value]

    @property
    def all_tasks(self) -> list[str]:
        if self._all_tasks is None:
            self.load_split()
        return self._all_tasks  # type: ignore[return-value]

    def get_group(self, task_id: str) -> str:
        """Return 'sft_train' or 'holdout' for a given task ID."""
        if task_id in self.train_tasks:
            return "sft_train"
        if task_id in self.holdout_tasks:
            return "holdout"
        return "unknown"

    def is_holdout(self, task_id: str) -> bool:
        return task_id in self.holdout_tasks

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #

    def _write_json(self, name: str, data: list[str]) -> None:
        path = self.split_dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Wrote %s (%d items)", path, len(data))

    def _read_json(self, name: str) -> list[str]:
        path = self.split_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"Split file not found: {path}. Run create_split() first."
            )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def ensure_split_exists(seed: int = 42) -> TaskSplit:
    """Return a TaskSplit, creating the split files if they don't exist."""
    ts = TaskSplit()
    split_file = ts.split_dir / "sft_train_tasks.json"
    if not split_file.exists():
        ts.create_split(seed=seed)
    else:
        ts.load_split()
    return ts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ensure_split_exists()
    ts = TaskSplit()
    ts.load_split()
    print(f"Train ({len(ts.train_tasks)}): {ts.train_tasks}")
    print(f"Holdout ({len(ts.holdout_tasks)}): {ts.holdout_tasks}")
    print(f"All ({len(ts.all_tasks)}): {ts.all_tasks}")
