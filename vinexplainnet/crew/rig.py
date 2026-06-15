from __future__ import annotations

import torch


def pick_device(prefer: str = "cpu") -> torch.device:
    choice = prefer.lower()
    if choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if choice == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
