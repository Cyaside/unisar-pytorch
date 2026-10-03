"""Optimizer/sampler/RNG recovery and inference-export contract tests."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from unisar.batch import collate_requests
from unisar.model import ModelConfig, UniSAR
from unisar.training import TrainConfig, TaskSampler, train, validate, load_export, export_checkpoint, microbatches


class TinyDataset:
    item_count = 10
    token_count = 10
    data_hash = "synthetic-training-contract-v1"

    def __init__(self):
        self.rows = []
        for task in ("search", "recommendation"):
            for i in range(4):
                positive = 2 + i % 2
                query = [positive]
                self.rows.append({"task": task, "item_ids": [2, 3, 4], "item_raw": [2, 3, 4],
                    "caption_ids": [[2], [3], [4]], "query_ids": query if task == "search" else [],
                    "query_tokens": query if task == "search" else [],
                    "labels": [int(x == positive) for x in (2, 3, 4)], "user_raw": str(i),
                    "request_id": f"{task}-{i}", "cohort": "transition", "timestamp_ms": 3000,
                    "history": [{"timestamp_ms": 1000, "service_mask": 2, "gap_bucket": 0,
                                 "rec_item_ids": [positive], "search_query_ids": []},
                                {"timestamp_ms": 2000, "service_mask": 1, "gap_bucket": 1,
                                 "rec_item_ids": [], "search_query_ids": [query]}]})

    def requests(self, split, task=None):
        assert split in ("train", "validation"), "Development must never read final test"
        return [r for r in self.rows if task is None or r["task"] == task]


def config(directory, **kwargs):
    value = TrainConfig(output_root=str(directory), run_id="fixture", steps=40, batch_size=4,
        validation_every=0, checkpoint_every=7, threads=1, max_seconds=120,
        lr=.01, alignment_weight=.01, transition_weight=.01,
        model={"embedding_dim": 8, "hidden_dim": 8, "heads": 2, "ff_dim": 16,
               "dropout": .1, "max_history": 4, "contrastive_mode": "infonce", "use_candidate_text": True})
    return replace(value, **kwargs)


def snapshot(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_interrupted_resume_exact_cpu(tmp_path):
    ds = TinyDataset()
    full_cfg = config(tmp_path / "full")
    full = train(ds, full_cfg)
    partial_cfg = config(tmp_path / "split", stop_after=17, validate_at_end=False)
    assert train(ds, partial_cfg)["completed_steps"] == 17
    resumed = train(ds, replace(partial_cfg, stop_after=None, validate_at_end=True, resume="latest"))
    a = snapshot(tmp_path / "full/fixture/latest.pt")
    b = snapshot(tmp_path / "split/fixture/latest.pt")
    assert a["step"] == b["step"] == 40
    for key in a["models"]:
        assert torch.equal(a["models"][key], b["models"][key]), key
    assert a["sampler"] == b["sampler"]
    assert a["scheduler"] == b["scheduler"]
    assert full["best_validation_joint_log_loss"] == resumed["best_validation_joint_log_loss"]


def test_learning_and_export_parity(tmp_path):
    ds = TinyDataset()
    cfg = config(tmp_path, steps=100)
    torch.manual_seed(cfg.seed)
    initial = UniSAR(ModelConfig(item_count=ds.item_count, token_count=ds.token_count, **cfg.model))
    initial_bce = validate(initial, ds.requests("validation"), cfg)[0]["joint_log_loss"]
    report = train(ds, cfg)
    assert report["best_validation_joint_log_loss"] < initial_bce * .8
    source = tmp_path / "fixture/best.pt"
    target = tmp_path / "exports/weights.pt"
    assert export_checkpoint(source, target)["bytes"] > 0
    model = load_export(target, ds.data_hash)
    original = snapshot(source)
    batch = collate_requests(ds.rows, model.max_history)
    expected = UniSAR(ModelConfig(**original["extra"]["architecture"]))
    expected.load_state_dict(original["models"])
    expected.eval()
    with torch.no_grad():
        assert torch.equal(model(batch), expected(batch))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_export(target, "different-dataset")
    payload = torch.load(target, weights_only=True)
    assert not {"optimizer", "rng", "sampler", "records"}.intersection(payload)


def test_recovery_rejects_changed_config_or_dataset(tmp_path, monkeypatch):
    ds = TinyDataset()
    cfg = config(tmp_path, steps=5, stop_after=2, validate_at_end=False)
    train(ds, cfg)
    with pytest.raises(FileExistsError):
        train(ds, replace(cfg, stop_after=None))
    with pytest.raises(ValueError, match="configuration hash mismatch"):
        train(ds, replace(cfg, lr=.002, resume="latest"))
    ds.data_hash = "changed"
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        train(ds, replace(cfg, resume="latest"))
    ds.data_hash = TinyDataset.data_hash
    import unisar.training as implementation
    original_provenance = implementation.software_provenance()
    monkeypatch.setattr(implementation, "software_provenance", lambda: {
        **original_provenance, "implementation_hash": "changed-code"})
    with pytest.raises(ValueError, match="configuration hash mismatch"):
        train(ds, replace(cfg, resume="latest"))


def test_sampler_and_complete_slates():
    ds = TinyDataset()
    sampler = TaskSampler(ds.requests("train", "search"), ds.requests("train", "recommendation"), 7)
    state = sampler.state_dict()
    expected = sampler.batch("recommendation", 20)
    sampler.load_state_dict(state)
    assert sampler.batch("recommendation", 20) == expected
    assert all(len(r["item_ids"]) == len(r["labels"]) == len(r["caption_ids"]) == 1 for r in expected)
    huge = {"item_ids": list(range(100))}
    tiny = {"item_ids": [2]}
    parts = list(microbatches([huge, tiny], 8))
    assert sum(len(x["item_ids"]) for p in parts for x in p) == 101


def test_invalid_paths_and_cpu_amp(tmp_path):
    for run_id in ("..", "../escape", "C:\\escape"):
        with pytest.raises(ValueError):
            config(tmp_path, run_id=run_id).check()
    with pytest.raises(ValueError, match="AMP"):
        config(tmp_path, amp=True).check()
