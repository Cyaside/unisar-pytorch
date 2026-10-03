# Data and information-time contract

The model consumes encoded requests, independent of TRACE. The supplied preparation workflow produces the same `prepared-blocks-v1` format that TRACE consumes. The [reproduction guide](reproduction.md) distinguishes this format from the original authors' processed datasets.

## Release and ownership

Use the existing public [KuaiSAR-small release](https://zenodo.org/records/8181109), file `KuaiSAR.zip`. Its published MD5 is `daea8cbf605db6bd5841740f0e4a12d9`. The [dataset project](https://kuaisar.github.io/) identifies the small release and states a CC BY-NC-SA 4.0 license. Data files are downloaded by the user and excluded from Git. Code attribution and dataset attribution are separate obligations.

No additional scraping, personal traffic or user recruitment is needed. Anonymous dataset users are the offline evaluation population. Results measure prediction on this historical short-video population; they do not estimate conversion or revenue at a travel, commerce or ride-hailing company.

## Canonical records

| Entity | Rule |
|---|---|
| Search request | Key `(user_raw, search_session_id)`; exactly one timestamp and query signature |
| Search candidate | Keep supported VIDEO/IMAGE_ATLAS candidates; retain missing metadata; deduplicate an item with maximum binary click |
| Recommendation | Keep observed distribution/exposure rows and their binary labels; repeated exposure rows remain separate targets |
| History block | One user and one timestamp, with pooled recommendation IDs and issued queries |
| Tied services | Simultaneous; do not invent directed order |

Search candidates are the logged candidates. No positive is injected into a sampled candidate list. A recommendation timestamp block is not asserted to be a verified slate; recommendation ranking within it is diagnostic.

## Cohort and split

Training eligibility requires at least one valid training search group and one training recommendation timestamp block. Sort eligible users by SHA-256 of UTF-8 `42:<raw_user_id>` and take a fixed prefix. Larger cohorts remain nested. Selecting users based on training eligibility does not inspect their eventual test activity.

The small-release protocol uses UTC+8, matching the release clock:

| Split | Interval |
|---|---|
| Training | Archive start through times strictly before 2023-05-29 00:00 |
| Validation | 2023-05-29 00:00 through times strictly before 2023-05-30 00:00 |
| Final test | 2023-05-30 00:00 through times strictly before 2023-05-31 00:00 |
| Unused trailing data | 2023-05-31 00:00 onward |

Every request at a boundary enters the later interval whole. Final test predictions must remain unavailable during training-readiness work, model selection and support-driven cohort decisions. Test count-only structural checks do not justify test metric tuning.

The release itself selected users who used both services across its collection window. That external population restriction remains even though downstream eligibility is training-only.

## Prediction inputs and feedback

For a request at time `t`, history must satisfy `history_timestamp < t`. Construct context before scoring the current timestamp block; append its summary after all current-block requests are scored. Earlier validation identities may enter later validation context during chronological replay, with fitted weights and vocabulary frozen.

The primary history contains:

- Recommendation distribution item identities under the declared log-availability assumption.
- Issued search query identities.
- Service presence and timestamps needed to preserve block chronology.

It excludes historical clicks, playback, likes, follows and search-result item IDs. A session-start timestamp does not prove that all search results or eventual feedback were already available. Current labels may be read by the training loss, including query/item auxiliary supervision, but never by the scoring function or input construction.

Current candidate captions use a declared static released-catalog assumption. Vocabulary fitting must not use future-only item captions. An ID-only sensitivity experiment is required before interpreting content benefits. A delayed-feedback extension would need its own availability rule, data hash, configuration and comparison; it is not enabled by the default workflow.

## Encoded request schema

| Field | Meaning |
|---|---|
| `task` | `search` or `recommendation` |
| `item_ids` | Nonempty candidate-aligned encoded identities; real candidates cannot be PAD |
| `item_raw` | Candidate-aligned original anonymous IDs, retained for tie rules and duplicate recognition |
| `caption_ids` | Candidate-aligned token sequences; empty caption is allowed |
| `query_ids` | Current search tokens; recommendation uses a learned replacement query |
| `query_tokens` | Original anonymized token identity sequence used to recognize duplicate queries |
| `history` | Chronological list of strictly earlier blocks |
| `labels` | Candidate-aligned binary targets; loss/evaluation only |
| `user_raw`, `request_id` | Stable identities for grouping, provenance and duplicate-negative exclusion |
| `timestamp_ms` | Request time used for prefix assertions |
| `cohort` | `transition`, `same_service` or `ambiguous`; evaluation metadata |

A history block has `timestamp_ms`, `rec_item_ids`, `search_query_ids`, `service_mask` and `gap_bucket`. Service bits are search=1, recommendation=2, simultaneous=3. Pool block contents independently of row order. Retain at most the last 50 blocks by default; token sequences are capped at 50 during preparation. Padding is introduced only by collation and cannot become a real item or event.

Vocabulary index 0 is PAD; index 1 is OOV. Item vocabulary fits training exposures/history. Token vocabulary fits training issued queries plus metadata from training-seen items. Raw IDs must remain distinct even when they share the OOV embedding; the OOV bucket is not an item identity for contrastive duplicate filtering.

The optional user embedding vocabulary, if explicitly enabled in a separate experiment, must be fit on training users and preserve a user OOV index. Its default disabled setting keeps controlled TRACE comparisons free of fitted user identity features.

## Prepared artifacts

```text
prepared/
  manifest.json
  requests.parquet
  blocks.parquet
  metadata.parquet
  vocabulary.json
  feature-policy.json
```

Requests store prefix pointers, not duplicated expanded history. Metadata and token lists retain raw identities for audit and deterministic re-encoding. The manifest records release, canonical, cohort, split, vocabulary and feature-policy hashes, together with file hashes and coverage counts. Loaders verify file hashes before use. Changed data, feature policy or vocabulary creates a new prepared directory and run; do not overwrite historical pilot artifacts.

## Transition support

The evaluation transition flag is determined from the immediately preceding strictly earlier block. That block must contain one service only, differ from the current request service, and fall within 30 minutes. A mixed previous block, no previous block, or longer gap is ambiguous. This flag is independent of click labels and is different from the architecture's four attention streams.

Search ranking eligibility requires at least two candidates and both a positive and a negative. Publish request and user denominators for overall and transition subsets. The later support check of 500 eligible transition requests and 100 users is a provisional minimum, not a statistical-power guarantee. When support is insufficient, retain the result as exploratory instead of changing the transition definition to improve metrics.

## Required invariants

1. Adding future records does not change earlier context or scores.
2. Changing current labels does not change input features or inference scores.
3. Permuting simultaneous rows does not change pooled history.
4. Extra padding does not change inference scores.
5. Empty histories, empty queries, OOV items and missing captions yield finite outputs.
6. Full candidate lists survive collation and scoring; chunking cannot change group loss normalization.
7. Resume requires matching data/configuration provenance and restores sampling and RNG state.
8. Validation and final test labels never become optimization targets or distillation cache entries.

## Evaluation interpretation

Search reports user-macro and request-macro NDCG@10 on eligible logged groups, with positive/negative/singleton exclusions. Stable score ties use ascending numeric raw item ID. All search candidates remain in log loss. Recommendation reports logged-exposure log loss, average precision and prevalence. Without known exposure propensities, these observations do not identify the value of a new serving policy or an online business outcome.
