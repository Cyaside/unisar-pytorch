from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from unisar.batch import collate_requests, to_device
from unisar.model import ModelConfig, SafeAttention, UniSAR, transition_masks
from unisar.losses import joint_loss


def examples():
    result = []
    for row, task in enumerate(("search", "recommendation", "search", "recommendation")):
        result.append({"task": task, "user_raw": f"u{row}", "query_tokens": [f"q{row}"],
                       "item_raw": [f"i{row * 2}", f"i{row * 2 + 1}"],
                       "item_ids": [2 + row * 2, 3 + row * 2], "labels": [1, 0],
                       "caption_ids": [[2, 3], [4]], "query_ids": [2 + row, 7],
                       "timestamp_ms": 50,
                       "history": [
                           {"timestamp_ms": 10, "service_mask": 2, "rec_item_ids": [2 + row], "gap_bucket": 0},
                           {"timestamp_ms": 20, "service_mask": 1, "search_query_ids": [[3 + row]], "gap_bucket": 1},
                           {"timestamp_ms": 30, "service_mask": 2, "rec_item_ids": [3 + row], "gap_bucket": 1},
                           {"timestamp_ms": 40, "service_mask": 1, "search_query_ids": [[4 + row]], "gap_bucket": 1},
                       ]})
    return result


def model(**kwargs):
    torch.manual_seed(42)
    return UniSAR(ModelConfig(32, 32, embedding_dim=8, hidden_dim=16, heads=2,
                             ff_dim=32, expert_hidden=16, dropout=0, **kwargs))


def test_empty_histories_and_padding_have_finite_outputs_and_gradients():
    requests = examples()
    requests[0]["history"] = []
    requests[1]["history"] = requests[1]["history"][:1]
    requests[0]["item_ids"] = requests[0]["item_ids"][:1]
    requests[0]["item_raw"] = requests[0]["item_raw"][:1]
    requests[0]["labels"] = [1]
    requests[0]["caption_ids"] = [[]]
    batch, ranker = collate_requests(requests), model(layers=2)
    output = ranker.forward_with_aux(batch)
    loss = joint_loss(output, batch)["total"]
    assert torch.isfinite(output["logits"]).all() and torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in ranker.parameters() if p.grad is not None)


def test_opposite_mask_excludes_same_service_future_padding_and_mixed_blocks():
    valid = torch.tensor([[True, True, True, True, False]])
    services = torch.tensor([[2, 1, 3, 2, 0]])
    masks = transition_masks(valid, services)
    expected = torch.zeros((1, 5, 5), dtype=torch.bool)
    expected[0, 1, 0] = True
    expected[0, 3, 1] = True
    assert torch.equal(masks["cross_allowed"], expected)
    assert masks["search_allowed"][0, 2, 1]
    assert masks["recommendation_allowed"][0, 2, 0]
    assert not masks["r2s"][0, 2] and not masks["s2r"][0, 2]
    generic = transition_masks(valid, services, opposite_only=False)
    assert generic["cross_allowed"][0, 3, 3]


def test_blocked_attention_cannot_read_blocked_values_and_has_zero_empty_rows():
    torch.manual_seed(1)
    attention = SafeAttention(8, 2, 0).eval()
    query, memory = torch.randn(1, 2, 8), torch.randn(1, 3, 8)
    allowed = torch.tensor([[[True, False, False], [False, False, False]]])
    reference = attention(query, memory, allowed)
    changed = memory.clone()
    changed[:, 1:] = 10000
    assert torch.equal(reference, attention(query, changed, allowed))
    assert torch.equal(reference[:, 1], torch.zeros_like(reference[:, 1]))


def test_causal_encoder_prefix_cannot_see_later_behavior():
    requests = examples()
    ranker = model(layers=2).eval()
    before = ranker.encode_context(collate_requests(requests))
    requests[0]["history"][-1]["search_query_ids"] = [[15, 16]]
    after = ranker.encode_context(collate_requests(requests))
    assert torch.equal(before["search_fused"][0, :3], after["search_fused"][0, :3])
    assert torch.equal(before["rec_fused"][0, :3], after["rec_fused"][0, :3])


def test_candidate_chunk_and_batch_padding_parity():
    batch, ranker = collate_requests(examples()), model(use_candidate_text=True).eval()
    full = ranker(batch)
    assert torch.allclose(full, ranker(batch, candidate_chunk_size=1), atol=1e-6, rtol=1e-6)
    for row, request in enumerate(examples()):
        single = ranker(collate_requests([request]))
        assert torch.allclose(full[row], single[0], atol=1e-6, rtol=1e-6)


def test_labels_never_change_inference_and_query_changes_search_score():
    batch, ranker = collate_requests(examples()), model().eval()
    reference = ranker(batch)
    changed = deepcopy(batch)
    changed["labels"] = 1 - changed["labels"]
    assert torch.equal(reference, ranker(changed))
    requests = examples()
    requests[0]["query_ids"] = [20, 21]
    assert not torch.allclose(reference[0], ranker(collate_requests(requests))[0])


def test_gradients_reach_encoders_fusion_query_gates_experts_and_temperatures():
    ranker, batch = model(use_candidate_text=True), collate_requests(examples())
    loss = joint_loss(ranker.forward_with_aux(batch), batch)["total"]
    loss.backward()
    for name in ("search_encoder", "rec_encoder", "mixed_encoder", "search_fusion_self",
                 "rec_fusion_self", "search_fusion_cross", "rec_fusion_cross", "gates",
                 "shared_experts", "task_experts", "relation_similarity", "search_similarity", "rec_similarity"):
        parameters = [p for p in getattr(ranker, name).parameters() if p.requires_grad]
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters), name
    assert ranker.token_embedding.weight.grad[2:8].abs().sum() > 0
    assert ranker.null_query.grad.abs().sum() > 0
    assert ranker.log_temperatures.grad.abs().sum() > 0


@pytest.mark.parametrize("task", ["search", "recommendation"])
def test_homogeneous_tasks_and_optional_ablation_switches(task):
    requests = [x for x in examples() if x["task"] == task]
    batch, ranker = collate_requests(requests), model()
    reference = ranker(batch)
    losses = joint_loss(ranker.forward_with_aux(batch), batch)
    assert all(x.ndim == 0 and torch.isfinite(x) for x in losses.values())
    assert losses["rec_transition_alignment"].abs() > 0
    for field, value in (("use_fusion", False), ("cross_mode", "global")):
        variant = UniSAR(replace(ranker.config, **{field: value}))
        variant.load_state_dict(ranker.state_dict())
        assert not torch.allclose(reference, variant(batch))


def test_data_contract_rejects_noncausal_history_and_preserves_unordered_blocks():
    requests = examples()
    requests[0]["history"][0]["rec_item_ids"] = [2, 3, 4]
    batch = collate_requests(requests)
    ranker = model().eval()
    reference = ranker(batch)
    requests[0]["history"][0]["rec_item_ids"].reverse()
    assert torch.allclose(reference, ranker(collate_requests(requests)), atol=1e-6)
    requests[0]["history"][0]["timestamp_ms"] = 50
    with pytest.raises(ValueError, match="precede"):
        collate_requests(requests)


def test_config_and_device_contract():
    for kwargs in ({"hidden_dim": 7}, {"max_history": 0}, {"contrastive_mode": "bad"}):
        with pytest.raises(ValueError):
            ModelConfig(20, 20, **kwargs)
    batch = to_device(collate_requests(examples()), "cpu")
    assert all(value.device.type == "cpu" for value in batch.values() if isinstance(value, torch.Tensor))
    with pytest.raises(ValueError, match="positive"):
        model()(batch, candidate_chunk_size=0)
