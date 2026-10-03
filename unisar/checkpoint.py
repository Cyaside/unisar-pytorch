"""Atomic, provenance-checked training snapshots, including stochastic state."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any], require_cuda: bool = False) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None:
        if not torch.cuda.is_available() and require_cuda:
            raise ValueError("CUDA RNG restoration requires CUDA; resume on the original device type.")
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save_checkpoint(path: str | Path, *, models, optimizer, scheduler, step: int,
                    sampler_state, config_hash: str, data_hash: str, config: dict,
                    scaler=None, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    snapshot = {
        "format_version": 1, "models": models.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": int(step), "sampler": sampler_state, "rng": rng_state(),
        "config_hash": config_hash, "data_hash": data_hash,
        "config": config, "extra": extra or {},
    }
    with temporary.open("wb") as handle:
        torch.save(snapshot, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, *, models, optimizer, scheduler,
                    config_hash: str, data_hash: str, scaler=None,
                    map_location="cpu", restore_random: bool = True,
                    expected_run_id: str | None = None) -> dict[str, Any]:
    # These are locally generated trusted snapshots, not downloaded model files.
    snapshot = torch.load(path, map_location=map_location, weights_only=False)
    if snapshot.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format.")
    if snapshot["config_hash"] != config_hash:
        raise ValueError("Resume configuration hash mismatch.")
    if snapshot["data_hash"] != data_hash:
        raise ValueError("Resume dataset hash mismatch.")
    if expected_run_id is not None and snapshot["config"].get("run_id") != expected_run_id:
        raise ValueError("Resume run identity mismatch; retain the checkpoint's original run ID.")
    models.load_state_dict(snapshot["models"])
    optimizer.load_state_dict(snapshot["optimizer"])
    scheduler.load_state_dict(snapshot["scheduler"])
    if scaler is not None and snapshot["scaler"] is not None:
        scaler.load_state_dict(snapshot["scaler"])
    if restore_random:
        restore_rng(snapshot["rng"], require_cuda=snapshot["config"].get("device") == "cuda")
    return snapshot
