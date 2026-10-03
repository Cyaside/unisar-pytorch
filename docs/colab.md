# Colab handoff

## Before opening

Use your own GitHub account with read access to `Cyaside/unisar-pytorch` and `Cyaside/TRACE`, and your own Google Drive for persistent files. Upload the notebook to Colab if the private GitHub import does not work. Select **Runtime → Change runtime type → GPU**. GPU allocation and quotas vary; no fixed allowance is assumed.

## Cell order

1. **Setup/authentication:** clone the pinned package and baseline TRACE revision, with model LFS downloads skipped. Enter a token through the hidden prompt. Install preparation pins and the package.
2. **Persistent storage:** mount Drive. The project folder holds prepared data, private run snapshots and exports. Raw/canonical intermediates use runtime disk.
3. **Download/verification:** obtain the public archive and verify size/MD5/SHA-256. Existing persistent prepared data allows raw download to be skipped.
4. **Cohort/vocabulary:** build the deterministic cohort and prepare features. Existing prepared data can be reused. The vocabulary is fitted on training only. Do not change split or transition rules to increase a validation metric.
5. **Preflight:** require a GPU, inspect available memory, run a bounded forward/backward check without optimizer updates, and record the device/resource report. If it fails, fix the input/configuration before training. For resource pressure, reduce batch size, padded-slot budget or candidate chunk size and give the revised experiment a new run ID; keep whole candidate groups.
6. **Teacher:** set `RUN_TEACHER_TRAINING = True` and run this cell yourself. The default false value makes Run all skip sustained training. The first invocation may stop at the time cap before the planned update budget.
7. **Recovery:** when `latest.pt` exists, leave the new-training gate false and set `RUN_RESUME_TRAINING = True` in the separate resume cell. Retain the run ID, planned steps, architecture, optimizer, data and AMP settings. Changing them is a new experiment. The checkpoint restores sampler, optimizer, scheduler and RNG state; exact CPU recovery is tested, GPU determinism is not promised.
8. **Export/validation:** after a best checkpoint exists, export it and compute validation outputs. Final test is reserved. Copying/exporting a partial run does not establish teacher quality.
9. **Optional matched baselines:** enable the separate baseline gate and run one selected shared/separate and input-policy configuration. Use identical planned budgets for comparisons. Query/caption controls retrain the selected policy. These longer runs are prepared here and still need execution.

## Compute and storage

The configuration uses FP32 initially, 32 search groups plus 32 sampled recommendation exposures per paired optimizer update, and a maximum of 10,000 updates. A session's time cap is checked between updates; checkpoint I/O and full validation can exceed it. Search slates remain complete, including a slate larger than the padding budget. Padding microbatches change contrastive negative pools and must be held fixed in matched comparisons.

Preflight estimates parameter and Adam-state memory and measures a real activation/gradient pass. It cannot promise peak memory for every rare slate or remaining Google quota. The embedding tables may dominate memory; attention compression alone does not solve this. Persistent storage needs prepared data plus latest/best snapshots and an exported model. Inspect free Drive space; do not delete the last recoverable checkpoint to make room. Checkpoint snapshots can be much larger than weights-only exports.

The small public archive is approximately 488 MB; extracted intermediates are several GB. Preparation runs on CPU even in a GPU runtime. Matching serializer pins avoids unnecessary cross-version differences. Reusing the prepared folder avoids repeating preparation after an interruption; the package verifies file hashes before training.

## After the run

Bring back `metrics.json`, `validation-best.json`, the preflight report and the exported weight file, together with the source revision/configuration/data hash. Keep snapshots and dated evidence private. Determine convergence, teacher/student quality and resource tradeoff before distillation. Transition subgroup support may be insufficient; preserve that limitation rather than claiming a conclusive improvement.

GPU preflight and sustained cloud training must be executed in your allocated Colab runtime. Local readiness checks do not prove those runs succeeded.
