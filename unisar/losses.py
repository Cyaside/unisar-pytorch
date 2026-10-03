"""Task-balanced BCE and duplicate-safe symmetric UniSAR contrastive losses."""
from __future__ import annotations

from typing import Any
import torch
from torch import Tensor
import torch.nn.functional as F


def paired_contrastive(left: Tensor, right: Tensor, identities: Tensor,
                       temperature: float | Tensor = .2, mode: str = "negative_only",
                       other_identities: Tensor | None = None) -> Tensor:
    """Paper's symmetric paired contrastive objective with safe eligibility.

    ``negative_only`` follows the printed Eq. 2/5 denominator, and may be
    negative. ``infonce`` includes the positive in the denominator. Every
    duplicate identity is excluded from negatives; without a distinct negative
    an anchor contributes zero rather than log(0). Inputs are already bilinear
    projected on the left; tanh implements the paper's bounded similarity.
    """
    if mode not in ("negative_only", "infonce"):
        raise ValueError("Unknown contrastive mode.")
    if isinstance(temperature, (int, float)) and temperature <= 0:
        raise ValueError("temperature must be positive.")
    zero = (left.float().sum() + right.float().sum()) * 0
    if left.shape != right.shape or identities.shape != (left.shape[0],):
        raise ValueError("Paired representations and identities must align.")
    if left.shape[0] < 2:
        return zero
    scores = torch.tanh(left.float() @ right.float().T) / temperature
    negatives = identities[:, None] != identities[None, :]
    if other_identities is not None:
        negatives &= other_identities[:, None] != other_identities[None, :]
    diagonal = torch.eye(len(left), dtype=torch.bool, device=left.device)
    values = []
    for matrix, allowed in ((scores, negatives), (scores.T, negatives.T)):
        eligible = allowed.any(1)
        if eligible.any():
            denominator = allowed | diagonal if mode == "infonce" else allowed
            # Slice before logsumexp: no hidden all--inf backward rows.
            selected = matrix[eligible].masked_fill(~denominator[eligible], -torch.inf)
            values.append((torch.logsumexp(selected, -1) - matrix.diagonal()[eligible]).mean())
    return torch.stack(values).sum() if values else zero


def supervised_loss(logits: Tensor, batch: dict[str, Any]) -> dict[str, Tensor]:
    """Equal mean of present tasks: search group means, rec exposure mean."""
    mask, task = batch["candidate_mask"], batch["task"]
    if logits.shape != mask.shape or batch["labels"].shape != mask.shape:
        raise ValueError("Logits, labels and candidate mask must align.")
    if not mask.any(1).all():
        raise ValueError("Every supervised request requires a real candidate.")
    values = F.binary_cross_entropy_with_logits(logits, batch["labels"], reduction="none")
    zero = logits.float().sum() * 0
    results, present = {}, []
    for code, name in ((0, "search"), (1, "recommendation")):
        selected = task == code
        if selected.any():
            if code == 0:
                loss = ((values[selected] * mask[selected]).sum(1) / mask[selected].sum(1).clamp_min(1)).mean()
            else:
                loss = values[selected][mask[selected]].mean()
            present.append(loss)
        else:
            loss = zero
        results[name] = loss
    if not present:
        raise ValueError("A loss batch must contain a recognized task.")
    results["supervised"] = torch.stack(present).mean()
    return results


def joint_loss(output: dict[str, Any], batch: dict[str, Any], alignment_weight: float = .1,
               transition_weight: float = .1, temperature: float = .2) -> dict[str, Tensor]:
    if temperature <= 0 or alignment_weight < 0 or transition_weight < 0:
        raise ValueError("Temperature must be positive; loss weights nonnegative.")
    result = supervised_loss(output["logits"], batch)
    mode = output.get("contrastive_mode", "negative_only")
    multipliers = output["temperature_multipliers"]
    lower, upper = output.get("temperature_bounds", (.03, 2.))
    temperatures = (temperature * multipliers).clamp(min=lower, max=upper)
    search = (batch["task"] == 0) & batch["query_known"]
    positive = search[:, None] & batch["candidate_mask"] & (batch["labels"] > 0) & (batch["item_ids"] > 1)
    rows, columns = positive.nonzero(as_tuple=True)
    # Multiple clicked items per query form pairs; duplicate query and item
    # identities cannot be false negatives. Average over pairs, not candidates.
    result["query_item_alignment"] = paired_contrastive(
        output["relation_queries"][rows], output["relation_items"][rows, columns],
        batch["query_identity"][rows], temperatures[0], mode,
        batch["candidate_identity"][rows, columns])
    for name, temp_index in (("search", 1), ("rec", 2)):
        valid = output[f"{name}_alignment_valid"]
        result[f"{name}_transition_alignment"] = paired_contrastive(
            output[f"{name}_left"][valid], output[f"{name}_right"][valid],
            batch["user_identity"][valid], temperatures[temp_index], mode)
    result["transition_alignment"] = result["search_transition_alignment"] + result["rec_transition_alignment"]
    result["total"] = (result["supervised"] + alignment_weight * result["query_item_alignment"]
                       + transition_weight * result["transition_alignment"])
    return result
