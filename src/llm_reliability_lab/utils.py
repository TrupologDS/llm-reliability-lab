"""Small shared utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object."""

    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        msg = f"Expected JSON object in {path}"
        raise ValueError(msg)
    return data


def write_json(data: Any, path: str | Path) -> Path:
    """Write JSON with stable formatting."""

    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=True, sort_keys=True)
        file.write("\n")
    return output_path


def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python and, if installed, NumPy and PyTorch."""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def none_if_blank(value: str | None) -> str | None:
    """Normalize blank strings from YAML/CLI arguments."""

    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
