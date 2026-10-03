"""Independent implementation of UniSAR's extraction, alignment and fusion.

Reference: Shi et al., SIGIR 2024, https://arxiv.org/abs/2404.09520.
The default heads follow paper Eq. 9 (shared and task-specific gated scalar
experts). This is not a copy of the upstream PLE implementation. TRACE's
timestamp blocks, query-only search histories and causal masking are explicit
adaptations; see the repository's paper mapping before claiming reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    item_count: int
    token_count: int
    embedding_dim: int = 32
    hidden_dim: int = 64
    heads: int = 2
    layers: int = 1
    ff_dim: int = 128
    dropout: float = .1
    max_history: int = 50
    user_count: int = 2
    shared_experts: int = 2
    task_experts: int = 2
    expert_hidden: int = 64
    use_candidate_text: bool = False
    causal: bool = True
    contrastive_mode: str = "negative_only"
    learnable_temperature: bool = True
    cross_mode: str = "opposite"
    use_fusion: bool = True
    min_temperature: float = .03
    max_temperature: float = 2.

    def __post_init__(self):
        for name in ("item_count", "token_count", "user_count"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must include PAD=0 and OOV=1.")
        for name in ("embedding_dim", "hidden_dim", "heads", "layers", "ff_dim",
                     "max_history", "shared_experts", "task_experts", "expert_hidden"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if self.contrastive_mode not in ("negative_only", "infonce"):
            raise ValueError("contrastive_mode must be negative_only or infonce.")
        if self.cross_mode not in ("opposite", "global"):
            raise ValueError("cross_mode must be opposite or global.")
        if not 0 < self.min_temperature <= self.max_temperature:
            raise ValueError("Temperature bounds must be positive and ordered.")


def _mean(values: Tensor, owners: Tensor, count: int, empty: Tensor | None = None) -> Tensor:
    result = values.new_zeros((count, values.shape[-1]))
    denominator = values.new_zeros((count, 1))
    if owners.numel():
        result.index_add_(0, owners, values)
        denominator.index_add_(0, owners, values.new_ones((len(owners), 1)))
    result = result / denominator.clamp_min(1)
    if empty is not None:
        result = torch.where(denominator > 0, result, empty[None, :])
    return result


def _pool(sequence: Tensor, valid: Tensor) -> Tensor:
    return (sequence * valid[..., None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]


class SafeAttention(nn.Module):
    """Hard masked attention with exactly zero output when no key is legal.

    A null row attends no real key. No future/padded key is inserted to avoid
    an all-masked softmax; this also avoids hidden NaNs in backward passes.
    """
    def __init__(self, dimension: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.head_dimension = heads, dimension // heads
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, memory: Tensor, allowed: Tensor) -> Tensor:
        b, q, d = query.shape
        k = memory.shape[1]
        def split(value: Tensor, length: int) -> Tensor:
            return value.view(b, length, self.heads, self.head_dimension).transpose(1, 2)
        queries = split(self.query(query), q)
        keys, values = split(self.key(memory), k), split(self.value(memory), k)
        # fp32 logits preserve numerical stability under autocast.
        scores = torch.matmul(queries.float(), keys.float().transpose(-2, -1)) / math.sqrt(self.head_dimension)
        legal = allowed[:, None, :, :]
        nonempty = legal.any(-1, keepdim=True)
        scores = scores.masked_fill(~legal, -torch.inf)
        scores = torch.where(nonempty, scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, -1) * legal
        attended = torch.matmul(self.dropout(weights).to(values.dtype), values)
        attended = attended.transpose(1, 2).reshape(b, q, d)
        return self.output(attended) * allowed.any(-1)[..., None]


class AttentionBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = SafeAttention(config.hidden_dim, config.heads, config.dropout)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.ff = nn.Sequential(nn.Linear(config.hidden_dim, config.ff_dim), nn.GELU(),
                                nn.Dropout(config.dropout), nn.Linear(config.ff_dim, config.hidden_dim))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query: Tensor, memory: Tensor, allowed: Tensor, valid: Tensor) -> Tensor:
        value = self.norm1(query + self.dropout(self.attention(query, memory, allowed)))
        value = self.norm2(value + self.dropout(self.ff(value)))
        return value * valid[..., None]


class SequenceEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([AttentionBlock(config) for _ in range(config.layers)])

    def forward(self, sequence: Tensor, allowed: Tensor, valid: Tensor) -> Tensor:
        for layer in self.layers:
            sequence = layer(sequence, sequence, allowed, valid)
        return sequence


def transition_masks(history_mask: Tensor, service_mask: Tensor, causal: bool = True,
                     opposite_only: bool = True) -> dict[str, Tensor]:
    """Three attention contracts over original timestamp-block positions.

    Same-service blocks include service=3 in both within-service streams.
    Cross-service edges require two unambiguous different services, excluding
    simultaneous mixed blocks because their internal order is unobserved.
    """
    h = history_mask.shape[1]
    search = history_mask & ((service_mask & 1) != 0)
    rec = history_mask & ((service_mask & 2) != 0)
    chronological = torch.ones((h, h), dtype=torch.bool, device=history_mask.device)
    if causal:
        chronological = chronological.tril()
    search_allowed = search[:, :, None] & search[:, None, :] & chronological
    rec_allowed = rec[:, :, None] & rec[:, None, :] & chronological
    single = history_mask & ((service_mask == 1) | (service_mask == 2))
    cross = single[:, :, None] & single[:, None, :] & (service_mask[:, :, None] != service_mask[:, None, :]) & chronological
    if not opposite_only:
        cross = history_mask[:, :, None] & history_mask[:, None, :] & chronological
    return {"search": search, "recommendation": rec, "search_allowed": search_allowed,
            "recommendation_allowed": rec_allowed, "cross_allowed": cross,
            "r2s": search & cross.any(-1), "s2r": rec & cross.any(-1)}


class UniSAR(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config, self.max_history = config, config.max_history
        e, d = config.embedding_dim, config.hidden_dim
        self.item_embedding = nn.Embedding(config.item_count, e, padding_idx=0)
        self.token_embedding = nn.Embedding(config.token_count, e, padding_idx=0)
        self.user_embedding = nn.Embedding(config.user_count, e, padding_idx=0)
        self.null_query = nn.Parameter(torch.empty(e).normal_(std=.02))
        self.search_input = nn.Linear(e, d)
        self.rec_input = nn.Linear(e, d)
        self.mixed_input = nn.Linear(e, d)
        self.positions = nn.ModuleDict({name: nn.Embedding(config.max_history, d)
                                       for name in ("search", "recommendation", "mixed")})
        self.search_encoder = SequenceEncoder(config)
        self.rec_encoder = SequenceEncoder(config)
        self.mixed_encoder = SequenceEncoder(config)
        # Eq. 7 includes another same-service self attention before fusion.
        self.search_fusion_self = AttentionBlock(config)
        self.rec_fusion_self = AttentionBlock(config)
        self.search_fusion_cross = AttentionBlock(config)
        self.rec_fusion_cross = AttentionBlock(config)
        self.search_target = nn.Linear(e, d, bias=False)
        self.rec_target = nn.Linear(e, d, bias=False)
        self.search_output = nn.Linear(d, e)
        self.rec_output = nn.Linear(d, e)
        def expert():
            return nn.Sequential(nn.Linear(5 * e, config.expert_hidden), nn.GELU(),
                                 nn.Dropout(config.dropout), nn.Linear(config.expert_hidden, 1))
        self.shared_experts = nn.ModuleList([expert() for _ in range(config.shared_experts)])
        self.task_experts = nn.ModuleList([nn.ModuleList([expert() for _ in range(config.task_experts)]) for _ in range(2)])
        self.gates = nn.ModuleList([nn.Linear(5 * e, config.shared_experts + config.task_experts) for _ in range(2)])
        self.relation_similarity = nn.Linear(e, e, bias=False)
        self.search_similarity = nn.Linear(d, d, bias=False)
        self.rec_similarity = nn.Linear(d, d, bias=False)
        self.log_temperatures = nn.Parameter(torch.zeros(3), requires_grad=config.learnable_temperature)

    def encode_context(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        valid, service = batch["history_mask"], batch["service_mask"]
        b, h = valid.shape
        if h > self.max_history:
            raise ValueError("History length exceeds model max_history.")
        masks = transition_masks(valid, service, self.config.causal, self.config.cross_mode == "opposite")
        rec = _mean(self.item_embedding(batch["history_rec_ids"]), batch["history_rec_owner"], b * h).view(b, h, -1)
        owners = batch["history_query_block"]
        query = _mean(self.token_embedding(batch["history_query_ids"]), batch["history_query_owner"], len(owners), self.null_query)
        search = _mean(query, owners, b * h).view(b, h, -1)
        def positioned(values: Tensor, mask: Tensor, name: str, projection: nn.Module) -> Tensor:
            # Within streams index positions by service count, not mixed gaps.
            positions = (mask.long().cumsum(1) - 1).clamp_min(0)
            return (projection(values) + self.positions[name](positions)) * mask[..., None]
        search_sequence = positioned(search, masks["search"], "search", self.search_input)
        rec_sequence = positioned(rec, masks["recommendation"], "recommendation", self.rec_input)
        mixed_sequence = positioned(search + rec, valid, "mixed", self.mixed_input)
        within_search = self.search_encoder(search_sequence, masks["search_allowed"], masks["search"])
        within_rec = self.rec_encoder(rec_sequence, masks["recommendation_allowed"], masks["recommendation"])
        # Preserve residual-only early blocks as keys across encoder layers.
        # Their extracted cross outputs remain ineligible until an opposite
        # predecessor exists; no future key is used to make them eligible.
        mixed_valid = valid & (service != 3) if self.config.cross_mode == "opposite" else valid
        cross = self.mixed_encoder(mixed_sequence, masks["cross_allowed"], mixed_valid)
        # Retain separate sequence outputs at each service's original positions.
        r2s, s2r = cross * masks["r2s"][..., None], cross * masks["s2r"][..., None]
        search_memory = self.search_fusion_self(within_search, within_search, masks["search_allowed"], masks["search"])
        rec_memory = self.rec_fusion_self(within_rec, within_rec, masks["recommendation_allowed"], masks["recommendation"])
        search_fused = self.search_fusion_cross(r2s, search_memory, masks["search_allowed"], masks["search"])
        rec_fused = self.rec_fusion_cross(s2r, rec_memory, masks["recommendation_allowed"], masks["recommendation"])
        if not self.config.use_fusion:
            search_fused, rec_fused = within_search, within_rec
        current_query = _mean(self.token_embedding(batch["query_ids"]), batch["query_owner"], b, self.null_query)
        current_query = torch.where((batch["task"] == 0)[:, None], current_query, self.null_query[None])
        return {"search_fused": search_fused, "rec_fused": rec_fused, "query": current_query,
                "search_mask": masks["search"], "rec_mask": masks["recommendation"],
                "search_left": self.search_similarity(_pool(within_search, masks["search"])),
                "search_right": _pool(r2s, masks["r2s"]),
                "rec_left": self.rec_similarity(_pool(within_rec, masks["recommendation"])),
                "rec_right": _pool(s2r, masks["s2r"]),
                "search_alignment_valid": masks["search"].any(1) & masks["r2s"].any(1),
                "rec_alignment_valid": masks["recommendation"].any(1) & masks["s2r"].any(1)}

    @staticmethod
    def _target_pool(sequence: Tensor, valid: Tensor, targets: Tensor) -> Tensor:
        scores = torch.einsum("bhd,bcd->bch", sequence.float(), targets.float())
        allowed = valid[:, None, :]
        nonempty = allowed.any(-1, keepdim=True)
        scores = scores.masked_fill(~allowed, -torch.inf)
        scores = torch.where(nonempty, scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, -1) * allowed
        return torch.einsum("bch,bhd->bcd", weights.to(sequence.dtype), sequence)

    def _score(self, batch: dict[str, Any], context: dict[str, Tensor], chunk_size: int | None) -> tuple[Tensor, Tensor]:
        b, c = batch["item_ids"].shape
        item = self.item_embedding(batch["item_ids"])
        candidates = item
        if self.config.use_candidate_text:
            captions = _mean(self.token_embedding(batch["caption_ids"]), batch["caption_owner"], b * c).view(b, c, -1)
            candidates = candidates + captions
        user_ids = batch.get("user_ids", batch["task"].new_ones(b))
        user = self.user_embedding(user_ids)
        chunk = c if chunk_size is None else int(chunk_size)
        if chunk < 1:
            raise ValueError("Candidate chunk size must be positive.")
        outputs = []
        for start in range(0, c, chunk):
            target = candidates[:, start:start + chunk]
            search = self.search_output(self._target_pool(context["search_fused"], context["search_mask"], self.search_target(target)))
            rec = self.rec_output(self._target_pool(context["rec_fused"], context["rec_mask"], self.rec_target(target)))
            # An empty history stays exactly zero despite output projection bias.
            search = search * context["search_mask"].any(1)[:, None, None]
            rec = rec * context["rec_mask"].any(1)[:, None, None]
            features = torch.cat([user[:, None].expand_as(target), target, context["query"][:, None].expand_as(target), search, rec], -1)
            shared = torch.cat([expert(features) for expert in self.shared_experts], -1)
            task_outputs = []
            for task in range(2):
                specific = torch.cat([expert(features) for expert in self.task_experts[task]], -1)
                experts = torch.cat([shared, specific], -1)
                task_outputs.append((self.gates[task](features).softmax(-1) * experts).sum(-1))
            logits = torch.stack(task_outputs, -1).gather(-1, batch["task"][:, None, None].expand(-1, target.shape[1], 1)).squeeze(-1)
            outputs.append(logits)
        return torch.cat(outputs, 1), item

    def forward(self, batch: dict[str, Any], candidate_chunk_size: int | None = None) -> Tensor:
        return self._score(batch, self.encode_context(batch), candidate_chunk_size)[0]

    def forward_with_aux(self, batch: dict[str, Any], candidate_chunk_size: int | None = None) -> dict[str, Any]:
        context = self.encode_context(batch)
        logits, item = self._score(batch, context, candidate_chunk_size)
        return {"logits": logits, "relation_queries": self.relation_similarity(context["query"]),
                "relation_items": item, "temperature_multipliers": self.log_temperatures.clamp(-4, 4).exp(),
                "temperature_bounds": (self.config.min_temperature, self.config.max_temperature),
                "contrastive_mode": self.config.contrastive_mode, **context}
