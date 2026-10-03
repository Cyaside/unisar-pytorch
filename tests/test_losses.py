import pytest
import torch
import torch.nn.functional as F

from unisar.batch import collate_requests
from unisar.model import ModelConfig, UniSAR
from unisar.losses import joint_loss, paired_contrastive, supervised_loss


def test_bce_task_balance_search_group_mean_rec_exposure_mean():
    logits = torch.tensor([[1., 2., 3.], [4., 5., 0.], [6., 7., 8.], [9., 0., 0.]])
    labels = torch.zeros_like(logits)
    mask = torch.tensor([[True, True, True], [True, True, False], [True, True, True], [True, False, False]])
    batch = {"labels": labels, "candidate_mask": mask, "task": torch.tensor([0, 0, 1, 1])}
    actual = supervised_loss(logits, batch)
    values = F.softplus(logits)
    search = (values[0].mean() + values[1, :2].mean()) / 2
    rec = torch.cat([values[2], values[3, :1]]).mean()
    assert torch.allclose(actual["search"], search)
    assert torch.allclose(actual["recommendation"], rec)
    assert torch.allclose(actual["supervised"], (search + rec) / 2)
    batch["candidate_mask"][3] = False
    with pytest.raises(ValueError, match="real candidate"):
        supervised_loss(logits, batch)


@pytest.mark.parametrize("mode", ["negative_only", "infonce"])
def test_paired_contrastive_matches_explicit_bilinear_tanh_formula(mode):
    left = torch.tensor([[1., 0.], [0., 1.], [.2, .3]])
    right = torch.tensor([[.5, .1], [.1, .6], [.4, .5]])
    scores = torch.tanh(left @ right.T) / .2
    terms = []
    for i in range(3):
        indexes = [j for j in range(3) if j != i or mode == "infonce"]
        terms.append(torch.logsumexp(scores[i, indexes], 0) - scores[i, i])
        terms.append(torch.logsumexp(scores[indexes, i], 0) - scores[i, i])
    expected = torch.stack(terms).sum() / 3
    actual = paired_contrastive(left, right, torch.arange(3), mode=mode)
    assert torch.allclose(actual, expected)


def test_duplicate_users_and_duplicate_items_are_never_negative():
    left = torch.randn(3, 4, requires_grad=True)
    right = torch.randn(3, 4, requires_grad=True)
    loss = paired_contrastive(left, right, torch.zeros(3, dtype=torch.long))
    assert loss == 0
    loss.backward()
    assert torch.equal(left.grad, torch.zeros_like(left))
    # Distinct queries that share the same positive item likewise give no
    # usable negative; this is not a negative query-item pair.
    loss = paired_contrastive(left, right, torch.arange(3), other_identities=torch.ones(3, dtype=torch.long))
    assert loss == 0


def test_single_anchor_and_empty_inputs_have_finite_zero_backward():
    for n in (0, 1):
        left, right = torch.randn(n, 3, requires_grad=True), torch.randn(n, 3, requires_grad=True)
        loss = paired_contrastive(left, right, torch.arange(n))
        assert loss == 0 and torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(left.grad).all()
    # fp16 reductions can overflow even when all individual inputs are finite.
    left = torch.full((1, 64), 10000., dtype=torch.float16, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    zero = paired_contrastive(left, right, torch.tensor([1]))
    assert zero == 0 and torch.isfinite(zero)
    zero.backward()
    assert torch.isfinite(left.grad).all()


def test_no_positive_or_oov_queries_items_cannot_create_query_item_alignment():
    requests = [{"task": "search", "item_ids": [1, 2], "labels": [1, 0],
                 "query_ids": [1], "user_raw": f"u{i}"} for i in range(2)]
    batch = collate_requests(requests)
    ranker = UniSAR(ModelConfig(5, 5, embedding_dim=4, hidden_dim=8, heads=2, ff_dim=16, dropout=0))
    loss = joint_loss(ranker.forward_with_aux(batch), batch)
    assert loss["query_item_alignment"] == 0
    assert loss["transition_alignment"] == 0
    loss["total"].backward()
    assert all(torch.isfinite(p.grad).all() for p in ranker.parameters() if p.grad is not None)


def test_zero_auxiliary_weights_recover_exact_supervised_loss():
    requests = [{"task": "search", "item_ids": [2 + i, 4], "labels": [1, 0],
                 "query_ids": [2 + i], "user_raw": f"u{i}"} for i in range(2)]
    batch = collate_requests(requests)
    ranker = UniSAR(ModelConfig(6, 6, embedding_dim=4, hidden_dim=8, heads=2, ff_dim=16, dropout=0))
    losses = joint_loss(ranker.forward_with_aux(batch), batch, alignment_weight=0, transition_weight=0)
    assert torch.equal(losses["total"], losses["supervised"])
    with pytest.raises(ValueError):
        joint_loss(ranker.forward_with_aux(batch), batch, temperature=0)


def test_temperature_is_learnable_but_bounded_for_extreme_parameter_values():
    requests = [{"task": "search", "item_ids": [2 + i, 4], "labels": [1, 0],
                 "query_ids": [2 + i], "user_raw": f"u{i}"} for i in range(2)]
    batch = collate_requests(requests)
    ranker = UniSAR(ModelConfig(6, 6, embedding_dim=4, hidden_dim=8, heads=2, ff_dim=16,
                               dropout=0, contrastive_mode="infonce"))
    for value in (-100., 100.):
        with torch.no_grad():
            ranker.log_temperatures.fill_(value)
        losses = joint_loss(ranker.forward_with_aux(batch), batch)
        assert all(torch.isfinite(x) for x in losses.values())
        ranker.zero_grad(set_to_none=True)
        losses["total"].backward()
        assert all(torch.isfinite(p.grad).all() for p in ranker.parameters() if p.grad is not None)
