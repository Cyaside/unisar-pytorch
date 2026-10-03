# Reproduction scope and training decisions

## What this repository establishes

It provides an independently written, importable UniSAR architecture and a resumable training workflow on the clean KuaiSAR-small protocol. Its source and tests make the implementation inspectable. Architecture correctness, numerical reproduction and teacher usefulness are different claims and require different evidence.

| Track | Purpose | Current scope |
|---|---|---|
| A: original numerical reproduction | Reproduce published results with original inputs, preprocessing, sampling and model choices | Unavailable until processed data and sampler parity are verified |
| B: clean logged-exposure study | Compare joint modeling and later compression with common causal features and logged candidates | Supplied runnable workflow |

Do not put Track A and B scores in one table as if they measured the same task.

## Original setting versus supplied setting

The original [paper](https://arxiv.org/html/2404.09520v1) filters sparse ending days and five-core interactions, uses feedback-enriched histories, and evaluates positives against 99 sampled negatives. Its prediction section names MMoE. The [authors' model](https://github.com/TengShi-RUC/UniSAR/blob/1a14a246066cd1aa6ee789c57ff2c7499f3d7379/models/UniSAR.py) uses PLE. These are recorded rather than silently reconciled.

Track B uses the [data contract](data-contract.md): fixed chronological boundaries; training-eligible nested cohorts; train-only vocabulary; unordered strictly prior timestamp blocks; historical recommendation distribution identities and issued queries; no historical feedback or search-result identities; retained logged candidates and OOV support. Its model follows the paper's gated shared/task-specific expert equation, with the explicit mask and contrastive interpretations in [paper mapping](paper-mapping.md).

The authors' [processed-data instructions](https://github.com/TengShi-RUC/UniSAR/blob/1a14a246066cd1aa6ee789c57ff2c7499f3d7379/data/README.md) expect separate search/recommendation train, validation and test pickle files plus item, query, search-session and user vocabularies. This is not proof of a byte-reproducible raw reconstruction. Their fixed embedding dimensions cannot be treated as the raw release's item/token counts.

## Gate before any Track A claim

1. Acquire the original processed release through its official instructions and record file hashes, permission notices and schema.
2. Verify filtering, split endpoints, historical feedback availability and exact user/item/query vocabularies.
3. Verify training sampling, excluded user interactions, 99-negative evaluation candidates, random seeds and candidate reuse.
4. Select and name paper-equation or code-PLE architecture. Record every framework port and mathematical interpretation.
5. Match objectives, task balance, reductions, temperature treatment, stopping rule and selection budget.
6. Reproduce evaluator fixtures and publish coverage differences before comparing headline scores.

Until these steps are satisfied, label a run as an architecture implementation or adapted study. Do not advertise verified SIGIR table reproduction.

## Commands for Track B

Run from the repository root after installing its package and dependencies. The standard development cohort is 10,000 training-eligible users; its name and immutable output directories make it distinct from earlier smaller cohorts.

```bash
python -m unisar.data.download --output data/raw
python -m unisar.data.audit --archive data/raw/KuaiSAR.zip --output data/canonical/preteacher-v1 --sizes 10000
python -m unisar.data.prepare --canonical data/canonical/preteacher-v1/10000 --output data/prepared/teacher-10000-v1
```

The commands download and verify the public archive, validate raw schema and construct the deterministic cohort, then fit training-only vocabulary and prefix pointers. They do not train a model. Preserve their manifests and use a new output directory if the cohort or policy changes.

Check the allocated runtime before training. The preflight verifies the prepared contract and runs a forward/backward check without creating an optimizer or updating weights. CUDA is required unless `--allow-cpu` is explicitly supplied for a local diagnostic; that option does not establish GPU readiness.

```bash
python -m unisar.preflight --data data/prepared/teacher-10000-v1 --config configs/teacher-colab.yaml --output /content/drive/MyDrive/TRACE-private/teacher-preflight.json
```

Training starts only when this command or its notebook cell is run explicitly. On Colab, use a persistent Drive output path rather than runtime-local storage:

```bash
python -m unisar.training train --data data/prepared/teacher-10000-v1 --config configs/teacher-colab.yaml --output-root /content/drive/MyDrive/TRACE-private/teacher-runs --run-id teacher-clean-v1 --device cuda
```

The starting configuration proposes 10,000 total optimizer updates, batch size 32 per task, learning rate 0.001, float32 and 50 history blocks. Its per-invocation time allowance is 3,000 seconds; time checks happen between updates, so validation/checkpoint I/O can extend wall time. This is a resumable development budget, not a promise of convergence or a quota guarantee.

Rerun the same command with `--resume latest` to restore its latest checkpoint. Keep the same run ID and experiment configuration; modifying operational time/checkpoint location is separate from changing the experiment. The Colab notebook contains the recommended preflight/profile cells before this manual training cell.

```bash
python -m unisar.training export --checkpoint /content/drive/MyDrive/TRACE-private/teacher-runs/teacher-clean-v1/best.pt --output /content/drive/MyDrive/TRACE-private/teacher-weights/teacher-clean-v1.pt
python -m unisar.training validate --data data/prepared/teacher-10000-v1 --weights /content/drive/MyDrive/TRACE-private/teacher-weights/teacher-clean-v1.pt --output /content/drive/MyDrive/TRACE-private/teacher-validation/teacher-clean-v1 --device cuda
```

Export `best.pt` only after full validation created it. `latest.pt` is the recovery checkpoint and need not be the selected model. The validation command evaluates validation only and does not unlock the final test.

## Training procedure for Track B

Use one fixed prepared cohort and vocabulary for teacher and baseline comparisons. Start with one seed and a bounded configuration; validation selects checkpoints. Each update samples one homogeneous search batch and one homogeneous recommendation batch, then gives each task half of the accumulated gradient. Search requests are sampled uniformly with replacement and retain their complete logged candidate groups. Recommendation exposure rows are sampled uniformly with replacement, with one exposure per sampled example. Current labels supervise stable logit BCE and training-only query/item relevance; transition alignment uses request histories.

The auxiliary query/item contrastive objective is expected to help distinguish query relevance, but this is a hypothesis until validation and masking ablations show a benefit. It cannot repair an item OOV rate by itself. Report ID/token coverage separately and preserve raw identities for duplicates.

Do not infer convergence from a short smoke run. Before longer training:

1. Confirm fixture loss decreases and inference ignores labels.
2. Profile an actual batch, candidate counts, peak memory and updates per second.
3. Check that both task losses receive nonzero gradients and that auxiliary losses have distinct negatives.
4. Estimate updates per exposure-equivalent pass from actual processed counts; balanced sampling can revisit sparse search groups more often than recommendation exposures.
5. Choose a bounded session duration, checkpoint frequency and persistent output directory.

Colab resource availability and quotas vary; the [official FAQ](https://research.google.com/colaboratory/faq.html) does not promise a fixed GPU or free-hour allowance. GPU compatibility, AMP behavior and cloud throughput must be checked on the allocated runtime before making cost claims. Local CPU correctness checks do not establish these GPU measurements.

## Loss coefficients and reductions

`joint_loss` exposes search/recommendation click losses, `query_item_alignment`, `search_transition_alignment`, `rec_transition_alignment` and their total. The configuration names `alignment_weight` for query/item relevance and `transition_weight` for the sum of the two transition alignments. Log the terms independently; a falling sum can conceal deterioration in one task.

For a homogeneous search microbatch, BCE is a mean over candidates within each request, then a mean over requests. For recommendation it is an exposure mean. If `B_s` and `B_r` are the two task batches, the update objective is:

```text
0.5 * total(B_s) + 0.5 * total(B_r)
total(B) = BCE(B)
         + alignment_weight * query_item_alignment(B)
         + transition_weight * [search_transition_alignment(B) + rec_transition_alignment(B)]
```

The relevance term is zero on recommendation batches. Query/item pairs are positive training clicks on known items and queries with some known tokens. Transition pairs require supported histories, and negatives must have different raw users. `negative_only` and `infonce` denominator modes are explicitly selectable; the base model's negative-only mode may have negative loss. The supplied training configuration explicitly chooses `infonce` for the adapted Track B run. Temperatures use the configured positive base times three learnable clamped multipliers, bounded by `min_temperature`/`max_temperature` (default 0.03/2.0).

To cap padding, each task batch is split into whole-request microbatches. The contribution of a fragment is weighted by its request count divided by the original task batch size. This preserves supervised task means but changes the available in-batch contrastive negative pool. Record `max_padded_candidates`, batch size and microbatch settings as part of the contrastive experiment; do not claim fragmentation-invariant contrastive loss.

Checkpoint selection minimizes full-validation `0.5 * search_group_log_loss + 0.5 * recommendation_exposure_log_loss`. Auxiliary loss is excluded from selection. AdamW with warmup/cosine scheduling and gradient clipping is the supplied training policy; these choices differ from the original paper's Adam/epoch stopping setting.

Search ranking candidates are kept complete. If scoring is chunked to fit memory, concatenate logits or accumulate the mathematically equivalent full-group reduction. Do not fit a different candidate sampler merely to make the teacher faster and then compare it with an unchanged baseline.

Train longer matched baselines alongside teacher experiments when earlier short runs have not established convergence. Use the same temporal features, data hashes, tuning opportunity and validation candidates. A teacher advantage measured under additional information or an altered item universe does not establish a joint-model advantage.

## Longer baseline and input-control workflow

`scripts/run_matched_baseline.py` prepares the remaining optimization and query/content checks as manually started cloud experiments. It uses the existing TRACE baseline implementation at revision `e93ab62af8491acdfc82dfd3054c17f329307d18`, rather than duplicating its shared/separate model code. The Colab notebook obtains that private source checkout with LFS smudge skipped, so prior large model exports are not downloaded. The script rejects a different revision or tracked source modifications.

Run shared and separate full-input screens first under the same total update budget. After inspecting their validation learning curves and resource use, run retrained input controls under that same declared budget. Available settings are:

| Option | Meaning |
|---|---|
| `--mode shared` | One joint baseline encoder and task-specific heads |
| `--mode separate` | Independent task-specific encoders with matching declared capacities |
| `--input-policy full` | Common current query, candidate caption and historical identity inputs |
| `--input-policy no_current_query` | Remove only the current query from training and validation; retain historical issued queries |
| `--input-policy no_caption` | Remove current candidate captions from training and validation; retain identities and queries |

These controls retrain the model with the input removed. They are different from removing an input only at evaluation. All controls preserve candidate IDs, labels, training vocabulary, chronological prefixes and complete logged groups. They cannot access a final-test view.

Example invocation from the standalone repository checkout, using the baseline source directory obtained in the notebook:

```bash
python scripts/run_matched_baseline.py --trace-root /content/TRACE --data data/prepared/teacher-10000-v1 --output-root /content/drive/MyDrive/TRACE-private/baseline-runs --mode shared --input-policy full --steps 10000 --batch-size 32 --max-seconds 3000 --device cuda
```

Use `--mode separate` for its matched comparator. Change `--input-policy` for a retrained control. Each setting gets a distinct default run ID such as `shared-full-10000-v1` or `separate-no_current_query-10000-v1`. To continue an interrupted setting, repeat its command with `--resume latest`. Do not change total steps, learning rate, model capacity, seed or validation frequency within a resumed experiment.

The defaults match the teacher's supervised sampling and selection contract: search-group-uniform and recommendation-exposure-uniform sampling with replacement; equal task weight per update; complete search groups; full-validation equal-task supervised BCE for checkpoint selection; batch size 32 per task; float32; learning rate 0.001; 10,000 planned total updates; 1,000-update validation frequency; 50 history blocks; candidate chunks of 256. This supports a controlled development comparison; it does not assert convergence or a fixed runtime.

The wrapper records the original prepared manifest and its new input policy separately in the ignored persistent run directory. Its effective data hash binds the original data hash and mask policy, so recovery cannot silently resume on different inputs. Underlying prepared files and vocabulary are not rewritten. Lightweight request proxies encode histories only when sampled or collated, including when the pinned trainer creates request lists; the existing bounded block cache remains in use.

Run one setting at a time after resource inspection. A soft session stop still triggers full validation and atomic checkpointing in the pinned trainer, which can extend wall time. The longer screens and retrained controls are ready to execute but their learning curves and conclusions remain pending until those cloud runs finish. Keep their metrics, predictions and resume bundles private.

## Checkpoints and exports

Persistent resume bundles contain model, optimizer, scheduler, scaler, sampler and RNG state with data/configuration provenance. Keep them outside Git. Resume the original run identity and configuration; a different model, data policy or selection rule creates a new run. Exact same-device CPU recovery is testable; cross-device floating-point equivalence is not promised.

The final weights export contains the architecture configuration and state dictionary for inference, with no optimizer or internal execution ledger. Exports and resume bundles have different purposes. Validate a reload against the selected model's logits before handoff. Keep labels and raw prediction evidence in ignored run output directories unless separately authorized for release.

## Ablations and later teacher gate

Suggested bounded screens answer distinct questions:

| Comparison | Question |
|---|---|
| `cross_mode=opposite` versus `global` | Does explicit opposite-service separation help compared with generic mixed-history attention? |
| Full model versus zero alignment coefficient | Does transition alignment help under identities-only histories? |
| Full model versus zero relevance coefficient | Does current query/item alignment improve query use? |
| `use_fusion=true` versus `false` | Is cross-attention needed after extraction? |
| Caption inputs versus ID-only | Is static catalog content useful under the declared snapshot assumption? |

Keep selection within the capped budget. The implementation-readiness endpoint does not include trained teacher superiority, three-seed final evaluation or distillation. After training, compare the teacher with compact hard-label S0 on identical candidates and one named CPU workload. Measure quality, embedding/encoder parameters, export bytes, peak memory and full-request latency. Proceed to distillation only when the teacher's measured quality/resource tradeoff makes the compression question useful.

## What to report

Every comparison needs protocol track, release/cohort/split/vocabulary/feature hashes, architecture interpretation, code revision, seed, tuning budget, task/group reductions, optimizer updates, processed search groups and recommendation exposures, precision, hardware, checkpoint selection, OOV coverage, metric denominators and unresolved assumptions. Keep private dated evidence outside the source repository. Release only explicitly approved aggregate results.
