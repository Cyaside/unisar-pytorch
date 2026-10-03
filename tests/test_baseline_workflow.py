"""Policy, laziness and manual invocation tests without baseline training."""
from copy import deepcopy
import importlib
from types import SimpleNamespace

import pytest

from scripts import run_matched_baseline as workflow
from unisar.common import read_json


class FakePrepared:
    item_count, token_count, data_hash = 5, 8, "fixture-prepared-v1"
    manifest = {"data_hash": data_hash, "feature_policy_hash": "identities-only-v1",
                "cohort_hash": "cohort-v1", "files": {"requests.parquet": "original-hash"}}

    def __init__(self, root=None):
        self.item_map = {100: 2, 200: 3}
        self.metadata = {100: {"caption_tokens": [9, 10]}, 200: {"caption_tokens": [11]}}
        self.history_calls = 0
        self.raw_requests = []
        for split in ("train", "validation", "test"):
            for task in ("search", "recommendation"):
                self.raw_requests.append({"split": split, "task": task, "user_raw": 7,
                    "request_id": f"{split}-{task}", "timestamp_ms": 30, "cohort": "transition",
                    "item_raw": [100, 200, 999], "labels": [1, 0, 0],
                    "query_tokens": [9, 10] if task == "search" else [], "history_ids": [0, 1]})

    def token_ids(self, raw):
        return [{9: 2, 10: 3, 11: 4}.get(token, 1) for token in raw]

    def encode_block(self, index):
        self.history_calls += 1
        return {"timestamp_ms": 10 + index * 10, "service_mask": 2 if index == 0 else 1,
                "rec_item_ids": [2] if index == 0 else [],
                "search_query_ids": [[2, 3]] if index else [], "gap_bucket": 0}


def test_list_materialization_keeps_prefixes_lazy_and_views_slice():
    base = FakePrepared()
    rows = list(workflow.InputPolicyDataset(base).requests("train"))
    assert base.history_calls == 0
    assert [row["task"] for row in rows] == ["search", "recommendation"]
    assert rows[0]["item_ids"] == [2, 3, 1]
    assert base.history_calls == 0
    assert len(rows[0]["history"]) == 2
    assert base.history_calls == 2
    view = workflow.InputPolicyDataset(base).requests("validation", "search")
    assert len(view[:1]) == 1 and view[0]["task"] == "search"


@pytest.mark.parametrize("policy", workflow.INPUT_POLICIES)
def test_policies_preserve_candidates_labels_history_and_base(policy):
    base = FakePrepared()
    original_rows, original_manifest = deepcopy(base.raw_requests), deepcopy(base.manifest)
    wrapper = workflow.InputPolicyDataset(base, policy)
    for split in ("train", "validation"):
        request = dict(wrapper.requests(split, "search")[0])
        assert request["item_raw"] == [100, 200, 999] and request["labels"] == [1, 0, 0]
        assert request["query_ids"] == ([] if policy == "no_current_query" else [2, 3])
        assert request["query_tokens"] == ([] if policy == "no_current_query" else [9, 10])
        assert request["caption_ids"] == ([[], [], []] if policy == "no_caption" else [[2, 3], [4], []])
        assert request["history"][1]["search_query_ids"] == [[2, 3]]
        assert request["history"][0]["rec_item_ids"] == [2]
        assert request["metadata_present"] == [True, True, False]
    assert base.raw_requests == original_rows and base.manifest == original_manifest
    assert wrapper.manifest["source_manifest"] == original_manifest


def test_policy_hashes_are_stable_distinct_and_final_test_is_inaccessible():
    hashes = {workflow.InputPolicyDataset(FakePrepared(), policy).data_hash for policy in workflow.INPUT_POLICIES}
    assert len(hashes) == 3
    assert workflow.InputPolicyDataset(FakePrepared()).data_hash == workflow.InputPolicyDataset(FakePrepared()).data_hash
    for bad in ("test", "unused", "future"):
        with pytest.raises(ValueError, match="final test is locked"):
            workflow.InputPolicyDataset(FakePrepared()).requests(bad)
    with pytest.raises(ValueError, match="input_policy"):
        workflow.InputPolicyDataset(FakePrepared(), "no_query")


def test_source_import_is_manual_and_pinned_revision_is_required(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda *a, **k: "different-revision\n")
    with pytest.raises(ValueError, match="revision mismatch"):
        workflow.load_trace_training(tmp_path)
    monkeypatch.setattr(workflow.subprocess, "check_output",
        lambda command, **k: workflow.TRACE_REVISION + "\n" if command[1] == "rev-parse" else " M src/models/neural.py\n")
    with pytest.raises(ValueError, match="tracked modifications"):
        workflow.load_trace_training(tmp_path)
    calls = []
    monkeypatch.setattr(workflow, "load_trace_training", lambda *a: calls.append(a))
    importlib.reload(workflow)
    assert calls == []


def test_verified_checkout_import_must_resolve_to_its_own_source(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(workflow.subprocess, "check_output",
        lambda command, **k: workflow.TRACE_REVISION + "\n" if command[1] == "rev-parse" else "")
    fake_config, fake_train = object(), object()
    module = SimpleNamespace(__file__=str(root / "src/training/neural.py"),
                             TrainConfig=fake_config, train=fake_train)
    calls = []
    def fake_import(name):
        calls.append(name)
        return module
    monkeypatch.setattr(workflow.importlib, "import_module", fake_import)
    monkeypatch.setattr(workflow.sys, "path", list(workflow.sys.path))
    assert workflow.load_trace_training(root) == (fake_config, fake_train)
    assert calls == ["src.training.neural"] and workflow.sys.path[0] == str(root)
    module.__file__ = str(root / "another-checkout/src/training/neural.py")
    with pytest.raises(RuntimeError, match="fresh Python process"):
        workflow.load_trace_training(root)


def test_manual_cli_dispatch_records_policy_and_equal_budget_without_training(tmp_path, monkeypatch):
    calls = []
    def fake_config(**kwargs):
        return SimpleNamespace(**kwargs)
    def fake_train(dataset, config):
        calls.append((dataset, config))
        assert len(dataset.requests("train")) == 2
        return {"status": "time_budget_reached", "completed_steps": 17}
    monkeypatch.setattr(workflow, "load_trace_training", lambda root: (fake_config, fake_train))
    import unisar.data.dataset as data_module
    monkeypatch.setattr(data_module, "PreparedDataset", FakePrepared)
    for mode in ("shared", "separate"):
        for policy in workflow.INPUT_POLICIES:
            result = workflow.main(["--trace-root", str(tmp_path), "--data", "fixture", "--output-root", str(tmp_path),
                                    "--mode", mode, "--input-policy", policy])
            assert result["completed_steps"] == 17
            dataset, config = calls[-1]
            assert config.steps == 10000 and config.batch_size == 32 and config.max_seconds == 3000
            assert config.validation_every == 1000 and config.device == "cuda"
            assert config.amp is False and config.record_ledger is False
            assert config.run_id == f"{mode}-{policy}-10000-v1"
            evidence = read_json(tmp_path / config.run_id / "input-policy.json")
            assert evidence["effective_data_hash"] == dataset.data_hash
            assert evidence["prepared_base_data_hash"] == "fixture-prepared-v1"
    workflow.main(["--trace-root", str(tmp_path), "--data", "fixture", "--output-root", str(tmp_path),
                   "--mode", "shared", "--resume", "latest"])
    assert calls[-1][1].resume == "latest"


def test_changed_policy_under_same_run_id_and_unsafe_names_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "load_trace_training", lambda root: (lambda **k: SimpleNamespace(**k),
        lambda *a: {"status": "complete", "completed_steps": 1}))
    import unisar.data.dataset as data_module
    monkeypatch.setattr(data_module, "PreparedDataset", FakePrepared)
    common = ["--trace-root", str(tmp_path), "--data", "fixture", "--output-root", str(tmp_path)]
    workflow.main(common + ["--run-id", "same"])
    with pytest.raises(ValueError, match="different input policy"):
        workflow.main(common + ["--run-id", "same", "--input-policy", "no_current_query"])
    for value in ("..", "../outside", "C:/outside", "a/b"):
        with pytest.raises(ValueError, match="safe single"):
            workflow.main(common + ["--run-id", value])
