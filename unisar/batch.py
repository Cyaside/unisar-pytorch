"""Ragged request contract with strictly earlier unordered-within-time histories.

This collator adapts our TRACE request contract. Simultaneous exposure blocks
remain unordered, and the target is logged-exposure BCE. Current outcomes are
carried for loss calculation and never become history features.
"""
from __future__ import annotations

from typing import Any, Sequence
from hashlib import sha256

import torch
from torch import Tensor


def _identity(value: Any) -> int:
    """Stable batch identity for duplicate-negative exclusion, never a feature."""
    return int.from_bytes(sha256(repr(value).encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


def _ids(values):
    values = [int(x) for x in values]
    if any(x < 0 for x in values):
        raise ValueError("Encoded identities must be nonnegative; PAD=0, OOV=1.")
    return [x for x in values if x != 0]


def collate_requests(examples: Sequence[dict[str, Any]], max_history: int = 50) -> dict[str, Any]:
    """Collate ragged identities without truncating candidates or block contents.

    Histories retain the last ``max_history`` *blocks*. Ragged token/identity
    owners avoid padding every simultaneous block to its largest item/query
    count. Labels are carried only for the loss and never read by the model.
    """
    if not examples or max_history < 1:
        raise ValueError("Need a nonempty batch and positive max_history.")
    b = len(examples)
    c = max(len(x["item_ids"]) for x in examples)
    if c == 0 or any(not x["item_ids"] for x in examples):
        raise ValueError("Every ranking request needs at least one candidate.")
    histories = [x.get("history", [])[-max_history:] for x in examples]
    h = max(1, max(len(x) for x in histories))
    batch: dict[str, Any] = {
        "item_ids": torch.zeros((b, c), dtype=torch.long),
        "candidate_mask": torch.zeros((b, c), dtype=torch.bool),
        "labels": torch.zeros((b, c), dtype=torch.float32),
        "history_mask": torch.zeros((b, h), dtype=torch.bool),
        "service_mask": torch.zeros((b, h), dtype=torch.long),
        "history_gap": torch.zeros((b, h), dtype=torch.long),
        "history_timestamp": torch.full((b, h), -1, dtype=torch.long),
        "request_timestamp": torch.tensor([int(x.get("timestamp_ms", -1)) for x in examples]),
        "request_gap": torch.tensor([int(x.get("request_gap", 0)) for x in examples]),
        "task": torch.tensor([0 if x["task"] == "search" else 1 for x in examples]),
        "user_ids": torch.tensor([int(x.get("user_id", 1)) for x in examples]),
        "user_identity": torch.tensor([_identity(x.get("user_raw", ("row", row))) for row, x in enumerate(examples)]),
        "query_identity": torch.tensor([_identity(x.get("query_tokens", x.get("query_ids", []))) for x in examples]),
        "candidate_identity": torch.zeros((b, c), dtype=torch.long),
        "query_known": torch.tensor([any(int(t) > 1 for t in x.get("query_ids", [])) for x in examples]),
        "raw": list(examples),
    }
    ragged = {name: [] for name in (
        "caption_ids", "caption_owner", "query_ids", "query_owner",
        "history_rec_ids", "history_rec_owner", "history_query_ids",
        "history_query_owner", "history_query_block",
    )}
    for row, (example, history) in enumerate(zip(examples, histories)):
        if example["task"] not in ("search", "recommendation"):
            raise ValueError("Unknown task.")
        items = [int(x) for x in example["item_ids"]]
        if any(x < 1 for x in items):
            raise ValueError("Actual candidates cannot be PAD or negative.")
        labels = example.get("labels", [0] * len(items))
        if len(labels) != len(items) or any(x not in (0, 1) for x in labels):
            raise ValueError("Candidate-aligned binary labels are required.")
        captions = example.get("caption_ids", [[] for _ in items])
        if len(captions) != len(items):
            raise ValueError("Captions must align with candidate identities.")
        batch["item_ids"][row, :len(items)] = torch.tensor(items)
        raw_items = example.get("item_raw", items)
        if len(raw_items) != len(items):
            raise ValueError("Raw candidate identities must align with item IDs.")
        batch["candidate_identity"][row, :len(items)] = torch.tensor([_identity(x) for x in raw_items])
        batch["candidate_mask"][row, :len(items)] = True
        batch["labels"][row, :len(items)] = torch.tensor(labels, dtype=torch.float32)
        for column, caption in enumerate(captions):
            ids = _ids(caption)
            ragged["caption_ids"].extend(ids)
            ragged["caption_owner"].extend([row * c + column] * len(ids))
        ids = _ids(example.get("query_ids", []))
        ragged["query_ids"].extend(ids)
        ragged["query_owner"].extend([row] * len(ids))
        previous_time = None
        for block_index, block in enumerate(history):
            owner = row * h + block_index
            service = int(block["service_mask"])
            if service not in (1, 2, 3):
                raise ValueError("History blocks must contain search, recommendation, or both.")
            timestamp = int(block.get("timestamp_ms", -1))
            if timestamp >= 0:
                if previous_time is not None and timestamp <= previous_time:
                    raise ValueError("History blocks must be strictly chronological.")
                request_time = int(example.get("timestamp_ms", -1))
                if request_time >= 0 and timestamp >= request_time:
                    raise ValueError("History must precede the current request.")
                previous_time = timestamp
            batch["history_mask"][row, block_index] = True
            batch["service_mask"][row, block_index] = service
            batch["history_gap"][row, block_index] = int(block.get("gap_bucket", 0))
            batch["history_timestamp"][row, block_index] = timestamp
            ids = _ids(block.get("rec_item_ids", []))
            ragged["history_rec_ids"].extend(ids)
            ragged["history_rec_owner"].extend([owner] * len(ids))
            for query in block.get("search_query_ids", []):
                query_index = len(ragged["history_query_block"])
                ragged["history_query_block"].append(owner)
                ids = _ids(query)
                ragged["history_query_ids"].extend(ids)
                ragged["history_query_owner"].extend([query_index] * len(ids))
    for name, values in ragged.items():
        batch[name] = torch.tensor(values, dtype=torch.long)
    if torch.any(batch["user_ids"] < 1):
        raise ValueError("Current users cannot be PAD or negative.")
    return batch


def to_device(batch: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}



