"""Run longer supervised TRACE baselines on the common prepared cohort.

This workflow requires the explicitly pinned TRACE source checkout. It does
not copy that architecture into the teacher package and never starts training
when imported. Run it manually after the Colab resource checks.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

# File invocation from a source checkout must find the adjacent package even
# before editable installation. Dependencies are still installed normally.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unisar.common import object_hash, read_json, write_json


TRACE_REVISION = "e93ab62af8491acdfc82dfd3054c17f329307d18"
INPUT_POLICIES = ("full", "no_current_query", "no_caption")
DERIVED_FIELDS = ("item_ids", "caption_ids", "metadata_present", "query_ids", "history")


def policy_definition(name: str) -> dict[str, Any]:
    if name not in INPUT_POLICIES:
        raise ValueError(f"input_policy must be one of {INPUT_POLICIES}")
    return {
        "version": "matched-baseline-input-policy-v1",
        "name": name,
        "current_query": name != "no_current_query",
        "current_candidate_captions": name != "no_caption",
        "historical_issued_queries": True,
        "historical_recommendation_ids": True,
        "candidate_or_label_changes": False,
        "fit": "same training-only item and token vocabulary for every policy",
    }


class PolicyRequest(Mapping):
    """A lightweight raw-row proxy that encodes only the accessed channel.

    The pinned trainer materializes request *lists*. Keeping these elements
    lazy avoids expanding every 50-block prefix at startup. Sampled requests
    and validation batches encode histories on demand. No encoded request is
    cached here, and masking does not modify raw rows or cached block content.
    """
    __slots__ = ("dataset", "raw", "policy")

    def __init__(self, dataset, raw, policy):
        self.dataset, self.raw, self.policy = dataset, raw, policy

    def __getitem__(self, key):
        if key == "query_tokens" and self.policy == "no_current_query":
            return []
        if key == "item_ids":
            return [self.dataset.item_map.get(item, 1) for item in self.raw["item_raw"]]
        if key == "caption_ids":
            if self.policy == "no_caption":
                return [[] for _ in self.raw["item_raw"]]
            return [self.dataset.token_ids(self.dataset.metadata.get(item, {}).get("caption_tokens", []))
                    for item in self.raw["item_raw"]]
        if key == "metadata_present":
            return [item in self.dataset.metadata for item in self.raw["item_raw"]]
        if key == "query_ids":
            return [] if self.policy == "no_current_query" else self.dataset.token_ids(self.raw["query_tokens"])
        if key == "history":
            return [self.dataset.encode_block(index) for index in self.raw["history_ids"]]
        return self.raw[key]

    def __iter__(self):
        yield from self.raw
        yield from (field for field in DERIVED_FIELDS if field not in self.raw)

    def __len__(self):
        return len(self.raw) + sum(field not in self.raw for field in DERIVED_FIELDS)


class PolicyView(Sequence):
    def __init__(self, dataset, indexes, policy):
        self.dataset, self.indexes, self.policy = dataset, indexes, policy

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return PolicyView(self.dataset, self.indexes[index], self.policy)
        return PolicyRequest(self.dataset, self.dataset.raw_requests[self.indexes[index]], self.policy)


class InputPolicyDataset:
    """Training/validation-only duck contract for the pinned baseline trainer."""
    def __init__(self, dataset, input_policy="full"):
        policy = policy_definition(input_policy)
        self.base, self.input_policy = dataset, input_policy
        self.item_count, self.token_count = dataset.item_count, dataset.token_count
        original = deepcopy(dataset.manifest)
        self.data_hash = object_hash({"base_data_hash": dataset.data_hash, "input_policy": policy})
        feature_hash = object_hash({"base_feature_policy_hash": original.get("feature_policy_hash"),
                                    "input_policy": policy})
        self.manifest = {**deepcopy(original), "source_manifest": original,
                         "base_data_hash": dataset.data_hash, "input_policy": policy,
                         "feature_policy_hash": feature_hash, "data_hash": self.data_hash}

    def requests(self, split, task=None):
        if split not in ("train", "validation"):
            raise ValueError("Matched baseline workflow permits train and validation only; final test is locked.")
        if task not in (None, "search", "recommendation"):
            raise ValueError("Unknown task")
        indexes = [i for i, row in enumerate(self.base.raw_requests)
                   if row["split"] == split and (task is None or row["task"] == task)]
        return PolicyView(self.base, indexes, self.input_policy)


def load_trace_training(trace_root: str | Path):
    """Verify a source-only TRACE checkout before importing its trainer."""
    root = Path(trace_root).resolve()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("--trace-root must point to the pinned TRACE Git checkout") from error
    if revision != TRACE_REVISION:
        raise ValueError(f"TRACE revision mismatch: expected {TRACE_REVISION}, found {revision}")
    changes = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no", "--", "src", "pyproject.toml"],
        cwd=root, text=True)
    if changes.strip():
        raise ValueError("Pinned TRACE source has tracked modifications; use its unchanged source checkout")
    sys.path.insert(0, str(root))
    module = importlib.import_module("src.training.neural")
    if Path(module.__file__).resolve() != root / "src" / "training" / "neural.py":
        raise RuntimeError("Another TRACE src package is already loaded; run this workflow in a fresh Python process")
    return module.TrainConfig, module.train


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--trace-root", required=True)
    result.add_argument("--data", required=True)
    result.add_argument("--output-root", required=True)
    result.add_argument("--mode", choices=("shared", "separate"), default="shared")
    result.add_argument("--input-policy", choices=INPUT_POLICIES, default="full")
    result.add_argument("--run-id")
    result.add_argument("--steps", type=int, default=10000)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--lr", type=float, default=.001)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--max-seconds", type=float, default=3000)
    result.add_argument("--validation-every", type=int, default=1000)
    result.add_argument("--checkpoint-seconds", type=float, default=180)
    result.add_argument("--candidate-chunk-size", type=int, default=256)
    result.add_argument("--max-padded-candidates", type=int, default=4096)
    result.add_argument("--resume", nargs="?", const="latest")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ("steps", "batch_size", "threads", "candidate_chunk_size", "max_padded_candidates"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.validation_every < 0 or min(args.lr, args.max_seconds, args.checkpoint_seconds) <= 0:
        raise ValueError("Invalid learning-rate, validation or session/checkpoint budget")
    run_id = args.run_id or f"{args.mode}-{args.input_policy}-{args.steps}-v1"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run_id must be a safe single directory name")
    TrainConfig, train = load_trace_training(args.trace_root)
    from unisar.data.dataset import PreparedDataset
    dataset = InputPolicyDataset(PreparedDataset(args.data), args.input_policy)
    output_root = Path(args.output_root).resolve()
    policy_path = output_root / run_id / "input-policy.json"
    contract = {"trace_revision": TRACE_REVISION, "run_id": run_id,
                "prepared_base_data_hash": dataset.base.data_hash,
                "effective_data_hash": dataset.data_hash, "manifest": dataset.manifest,
                "final_test": "locked; inaccessible through this wrapper"}
    if policy_path.exists():
        if read_json(policy_path) != contract:
            raise ValueError("Existing run has a different input policy or dataset; choose a new run_id")
    else:
        write_json(policy_path, contract)
    config = TrainConfig(run_id=run_id, mode=args.mode, steps=args.steps, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed, device=args.device, threads=args.threads,
        max_seconds=args.max_seconds, validation_every=args.validation_every,
        checkpoint_seconds=args.checkpoint_seconds, resume=args.resume, output_root=str(output_root),
        candidate_chunk_size=args.candidate_chunk_size, max_padded_candidates=args.max_padded_candidates,
        amp=False, record_ledger=False,
        question=f"Longer matched supervised baseline screen; input policy={args.input_policy}; no convergence assumption.")
    report = train(dataset, config)
    print(json.dumps({"run_id": run_id, "status": report["status"],
                      "completed_steps": report["completed_steps"], "input_policy": args.input_policy,
                      "output_root": str(output_root)}), flush=True)
    return report


if __name__ == "__main__":
    main()
