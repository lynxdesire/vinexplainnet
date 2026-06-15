from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    _atomic_write(Path(path), json.dumps(obj, indent=2, sort_keys=True).encode("utf-8"))


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def save_take(path: str | Path, state: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".pt")
    os.close(fd)
    try:
        torch.save(state, tmp)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_take(path: str | Path) -> dict[str, Any]:
    state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    return state
