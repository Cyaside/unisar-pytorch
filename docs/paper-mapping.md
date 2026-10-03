# Architecture and paper mapping

This repository independently implements the transition extraction, alignment and fusion design of [UniSAR, SIGIR 2024](https://doi.org/10.1145/3626772.3657811). The [full paper](https://arxiv.org/html/2404.09520v1) is the architectural reference. The default runnable workflow is an explicitly adapted logged-exposure study; see [data contract](data-contract.md) and [reproduction status](reproduction.md).

## Component map

| Paper reference | Mechanism | Package component |
|---|---|---|
| §4.1, Eq. 1 | ID/query representations and history construction | Embeddings and service-specific history pooling in `unisar/model.py` |
| Eq. 2 | Bidirectional query/item alignment | Relevance contrastive loss in `unisar/losses.py` |
| Eq. 3 | Separate search and recommendation encoders | `UniSAR` search/recommendation sequence encoders |
| Eq. 4 | Opposite-service masked mixed encoder | Mixed encoder and transition masks |
| Eq. 5–6 | Same-user transition alignment | Search and recommendation alignment losses |
| Eq. 7 | Self-attention followed by cross-attention fusion | Search/recommendation fusion modules |
| Eq. 8 | Candidate-conditioned history aggregation | Separate bilinear target attention modules |
| Eq. 9 | Shared and task-specific gated experts | Two expert gates and task heads |
| Eq. 10–12 | Joint click and auxiliary objectives | Stable logit BCE and weighted loss composition |

For a notation key, `s2s`/`r2r` denote within-service streams; `r2s`/`s2r` denote opposite-service information gathered at search/recommendation positions. They are learned representations, not explicit labels for an individual observed service switch.

## Computation implemented here

The following is the package's computation contract. It also identifies the changes needed for timestamp-block data.

### 1. Represent identities, queries and candidates

Item and token tables reserve index 0 for PAD and 1 for OOV. Masked token pooling makes the representation independent of additional PAD entries. The base model uses item IDs; `use_candidate_text=true` adds masked mean caption embeddings as an explicit content adaptation. Target projections put candidates in the sequence representation space. A query uses the same token table. Recommendation prediction uses a learned replacement query. The default user table has PAD and one shared OOV vector; it does not fit personal identity embeddings.

A history block has a pooled recommendation identity representation and a pooled issued-query representation. A block with both services contributes to both service streams. No clicked search result, historical label or feedback field enters this construction. This differs from the paper's feedback-enriched search behavior representation. Query/item relevance is an auxiliary training loss and may read current training labels; prediction inputs may not read them.

### 2. Extract four streams

Let `S` and `R` be compacted search and recommendation subsequences of the prior blocks; let `U` be the mixed stream. Each stream receives positional information. Search and recommendation have independent sequence encoders:

```text
A_s = search_encoder(S)
A_r = recommendation_encoder(R)
M   = mixed_encoder(U, opposite_service_mask)
B_s = M gathered at search positions
B_r = M gathered at recommendation positions
```

The runnable timestamp-block adapter also preserves strict chronology within the prefix. Its opposite-service mask requires two unambiguous single-service blocks and excludes keys at the same or a later block time. Mixed simultaneous blocks enter both within-service streams but are excluded from cross-service edges. A key must be a real history element and have an allowed source service. Empty attention rows produce a finite zero attention contribution; padding cannot become a surrogate history event.

The paper displays mask multiplication on attention logits, but describes exclusive opposite-service attention. Multiplying an excluded logit by zero still gives it a nonzero softmax weight. This package implements actual exclusion before normalization and records the mathematical interpretation as a deviation.

### 3. Align transitions during training

Masked means of `A_s` and `B_s` form the search pair; masked means of `A_r` and `B_r` form the recommendation pair. A pair is eligible only when its required histories exist. Other users provide negatives. Repeated requests from one user are excluded as negatives rather than treated as contradictory training examples.

The contrastive similarity is a trainable bilinear map followed by `tanh`; its temperature is positive. `contrastive_mode="negative_only"` follows the printed negative-only denominator and can produce a negative loss. `contrastive_mode="infonce"` instead includes the positive and yields a nonnegative cross-entropy objective. Record the selected mode with results; the two objectives are not interchangeable. Each direction uses mean eligible-anchor loss, and the two directions are summed. One eligible pair without any distinct negative contributes zero.

Query/item pairs use clicked candidates of training search requests only when the query has at least one known token and the item ID is known. Raw query/item identities are retained for duplicate exclusion; OOV-only inputs still participate in supervised prediction. Three learnable temperature multipliers scale the configured base temperature; their log values are clamped to `[-4,4]`, and effective temperatures are bounded by `min_temperature`/`max_temperature` (default 0.03/2.0). The supplied training configuration explicitly selects `infonce` for Track B. A negative-only auxiliary loss going below zero does not imply better validation ranking.

Contrastive terms act on request representations, not on a separate copied history for every candidate. This avoids giving large candidate groups an unintended additional alignment weight.

### 4. Fuse same-service and cross-service information

Each within-service stream first receives another self-attention operation. Its result provides keys and values; the corresponding opposite-service stream provides cross-attention queries. Feed-forward transformation completes each fusion branch:

```text
F_s = self_attention(A_s)
V_s = feed_forward(cross_attention(query=B_s, key=F_s, value=F_s))
F_r = self_attention(A_r)
V_r = feed_forward(cross_attention(query=B_r, key=F_r, value=F_r))
```

This extra self-attention belongs to fusion. It must not be removed merely because extraction already contains self-attention.

### 5. Score each supplied candidate

Two candidate-conditioned bilinear attention modules pool `V_s` and `V_r`. Concatenate the resulting search/recommendation context with user, candidate and current-query representations. A search gate mixes its own experts with shared experts; a recommendation gate mixes its own experts with the same shared expert set. The package returns logits, with sigmoid applied only when probabilities are required. This is a reranker of supplied candidates. Candidate-conditioned pooling does not directly supply one fixed user vector suitable for an ANN retrieval index.

The paper names this mechanism MMoE while its equation includes task-specific experts. Our default follows that equation. The inspected authors' source instead instantiates `PLE_layer`, a separate architectural choice. We do not claim our default is a port of that PLE implementation.

## Interpretation and adaptation register

| Choice in this package | Why it is explicit |
|---|---|
| Identities-only, unordered timestamp-block histories | Raw session/distribution times do not certify historical feedback availability |
| Strict prior-block cross-service attention | Avoids invented order among simultaneous service actions |
| OOV support and train-only vocabulary | Keeps future identities/captions out of fitting |
| Optional user embedding disabled by default | TRACE's controlled feature policy starts without personal identity features |
| Explicit `negative_only` and `infonce` modes | Distinguishes the printed objective from positive-inclusive cross-entropy |
| Duplicate user/query/item exclusions | Avoids false negatives in repeated requests and OOV buckets |
| Stable zero handling for empty pools/attention | Required for cold or single-service prefixes |
| Logits and declared task/group reductions | Prevents numerical overflow and accidental candidate-count weighting |
| Logged candidates and exposure BCE | Answers a different evaluation question from sampled next-item ranking |
| Two shared/two task experts by default | A bounded starting configuration rather than the paper's four-expert setting |
| Embedding width 32, sequence width 64 | Separate item/token and sequence capacities, explicitly recorded in configuration |

Architecture ablations can switch `cross_mode="global"` to replace opposite-service separation with a generic mixed-history encoder, or `use_fusion=false` to bypass cross-attention fusion. Zeroing the two auxiliary coefficients isolates alignment and relevance supervision. These answer different questions; global mixing does not remove cross-service history entirely. Report the actual setting and preserve common candidates/features. A successful implementation test proves a contract; teacher quality still requires training and held-out validation.

## Source provenance and attribution

The authors' implementation was inspected at [revision `1a14a246066cd1aa6ee789c57ff2c7499f3d7379`](https://github.com/TengShi-RUC/UniSAR/tree/1a14a246066cd1aa6ee789c57ff2c7499f3d7379), including its [model](https://github.com/TengShi-RUC/UniSAR/blob/1a14a246066cd1aa6ee789c57ff2c7499f3d7379/models/UniSAR.py), [layers](https://github.com/TengShi-RUC/UniSAR/blob/1a14a246066cd1aa6ee789c57ff2c7499f3d7379/models/layers.py) and [configuration](https://github.com/TengShi-RUC/UniSAR/blob/1a14a246066cd1aa6ee789c57ff2c7499f3d7379/config/UniSAR_KuaiSAR.yaml). That tree contains no explicit license file and its GitHub license metadata is null. Inspection is provenance, not permission to copy its source. The package implementation is independently written; upstream files are neither vendored nor redistributed.

Use [CITATION.cff](../CITATION.cff) to credit the original research. Do not describe UniSAR itself as this repository's novel algorithm.
