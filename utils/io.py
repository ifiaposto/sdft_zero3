"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================

Atomic file writes for EMA teacher checkpoints, checkpoint metadata, and evaluation JSON files.

Writes to <path>.tmp before atomically replacing the destination, preserving any existing
file if the process is interrupted during writing.
"""

from __future__ import annotations

import json
import os
from typing import Any


def atomic_write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2))


def atomic_torch_save(payload: Any, path: str) -> None:
    import torch

    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)

