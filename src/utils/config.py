"""YAML config loader with environment variable interpolation."""

import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def _interpolate_env(value: Any) -> Any:
    """Replace ${VAR} patterns with environment variable values."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(
            lambda m: os.environ.get(m.group(1), ""), value
        )
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def load_config(path: str | Path) -> dict:
    """Load a YAML config file with environment variable interpolation.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed config dict with env vars interpolated.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return _interpolate_env(raw)


def save_config(config: dict, path: str | Path) -> None:
    """Save a config dict to YAML (for frozen config snapshots)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def resolve_model_path(model_name: str) -> str:
    """Resolve a HuggingFace model ID to a local path on AutoDL.

    On AutoDL, models are downloaded to $AUTODL_TMP/models/<basename>.
    For example, "Qwen/Qwen2.5-7B-Instruct" resolves to
    "/root/autodl-tmp/models/Qwen2.5-7B-Instruct" if that directory exists.

    If the input is already a valid local path, or if no local directory
    is found, the original string is returned unchanged.
    """
    # Already a valid local path with config.json
    p = Path(model_name)
    if p.exists() and (p / "config.json").exists():
        return model_name

    # Try resolving under AUTODL_TMP/models/
    autodl_tmp = os.environ.get("AUTODL_TMP", "/root/autodl-tmp")
    basename = model_name.split("/")[-1]  # "Qwen/Qwen2.5-7B-Instruct" -> "Qwen2.5-7B-Instruct"
    local_path = Path(autodl_tmp) / "models" / basename
    if local_path.exists() and (local_path / "config.json").exists():
        return str(local_path)

    # Fall back to original (may work online or from HF cache)
    return model_name
