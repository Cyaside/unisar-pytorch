# UniSAR in PyTorch

Independent implementation of **UniSAR: Modeling User Transition Behaviors between Search and Recommendation** (Shi et al., SIGIR 2024). [Paper](https://arxiv.org/abs/2404.09520) · [Authors' implementation](https://github.com/TengShi-RUC/UniSAR)

The model extracts same-service and cross-service history signals with masked Transformers, aligns their representations, fuses them with cross-attention, and scores candidates through shared and task-specific gated experts. This repository also supplies a raw KuaiSAR preparation pipeline and a resumable training workflow.

## What is implemented

- Independent architecture, contrastive losses, architecture ablations and task-specific scoring.
- Chronological train/validation/test partitions, training-only vocabulary and strictly earlier timestamp-block context.
- Logged-candidate training, full validation, checkpoint recovery and weights-only export.
- Synthetic correctness/learning fixtures and a manually started Colab notebook.

The supplied workflow is a **clean-protocol adaptation**, not a verified reproduction of the paper's numerical results. Search histories use previously issued queries; historical feedback and search-result identities are omitted. The recommended configuration adds candidate captions and uses positive-inclusive InfoNCE. The paper's printed negative-only denominator is also available. See the [paper mapping](docs/paper-mapping.md), [data contract](docs/data-contract.md) and [reproduction scope](docs/reproduction.md).

## Colab

Upload [notebooks/train_teacher.ipynb](notebooks/train_teacher.ipynb) to [Google Colab](https://colab.research.google.com/), select a GPU runtime, and follow the numbered cells. A GitHub token with read access to the private repositories is entered through a hidden prompt; it is not stored in the notebook or remote URL.

The notebook downloads the public archive, prepares a 10,000-user training-eligible cohort, checks a real forward/backward pass and memory, and persists checkpoints to your Drive. **Training gates default to false.** Enable the teacher gate only after preflight, then run that cell yourself. See the [Colab handoff](docs/colab.md).

## Local setup and commands

Python 3.11 or newer within the supported range is required.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements-data.txt -e '.[dev]'
python -m pytest -q
python -m unisar.data.download --output data/raw
python -m unisar.data.audit --archive data/raw/KuaiSAR.zip --output data/canonical/development --sizes 10000
python -m unisar.data.prepare --canonical data/canonical/development/10000 --output data/prepared/teacher-10000-v1
python -m unisar.preflight --data data/prepared/teacher-10000-v1 --config configs/teacher-colab.yaml --output .local/preflight.json --allow-cpu
```

Training is an explicit separate command. Select a persistent `--output-root` and retain the same configuration when resuming:

```bash
python -m unisar.training train --data data/prepared/teacher-10000-v1 --config configs/teacher-colab.yaml --output-root artifacts/runs
python -m unisar.training train --data data/prepared/teacher-10000-v1 --config configs/teacher-colab.yaml --output-root artifacts/runs --resume latest
python -m unisar.training export --checkpoint artifacts/runs/teacher-clean-v1/best.pt --output artifacts/models/teacher/weights.pt
python -m unisar.training validate --data data/prepared/teacher-10000-v1 --weights artifacts/models/teacher/weights.pt --output artifacts/validation
```

The default development budget is 10,000 paired optimizer updates with up to 50 minutes per invocation. It is a starting budget, not a convergence claim or a promise about GPU hours. Full validation/checkpoint I/O can extend the invocation. Measure throughput before expanding experiments. [Colab quotas are variable](https://research.google.com/colaboratory/faq.html).

## Storage and attribution

Datasets, outputs, credentials and dated execution evidence are ignored. Download references and code replace dataset uploads. Training snapshots include optimizer/RNG state; exports contain architecture, tensors and contract hashes, without those snapshots. Rebuild the matching vocabulary before inference.

The official KuaiSAR project states [CC BY-NC-SA 4.0](https://github.com/KuaiSAR/KuaiSAR.github.io/blob/main/LICENSE). Dataset, paper, implementation and model-weight permissions are separate. This code was independently authored; the upstream code's reuse license was not verified. See [attribution](docs/attribution.md) and [CITATION.cff](CITATION.cff).

Teacher quality, GPU stability, numerical reproduction, teacher viability against a compact model and distillation results require subsequent training/evaluation. Final test scoring is intentionally absent from the development commands.

