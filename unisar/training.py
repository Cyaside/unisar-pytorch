"""Explicit, bounded training of the clean logged-candidate UniSAR adaptation."""
from __future__ import annotations

import argparse
import bisect
from dataclasses import asdict, dataclass, field
import json
import hashlib
import math
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any

import numpy as np
import psutil
import torch
import yaml

from unisar.batch import collate_requests, to_device
from unisar.checkpoint import load_checkpoint, save_checkpoint
from unisar.common import digest, object_hash, read_json, write_json
from unisar.losses import joint_loss
from unisar.model import ModelConfig, UniSAR
from unisar.metrics import evaluate_search, evaluate_recommendation


@dataclass
class TrainConfig:
    run_id: str = "teacher-clean-v1"
    steps: int = 10000
    batch_size: int = 32
    lr: float = .001
    weight_decay: float = .0001
    seed: int = 42
    device: str = "cpu"
    threads: int = 2
    max_seconds: float | None = 3000
    validation_every: int = 1000
    checkpoint_every: int = 100
    checkpoint_seconds: float = 180
    max_padded_candidates: int = 4096
    candidate_chunk_size: int = 256
    amp: bool = False
    alignment_weight: float = .1
    transition_weight: float = .1
    temperature: float = .2
    model: dict = field(default_factory=dict)
    output_root: str = "artifacts/runs"
    resume: str | None = None
    stop_after: int | None = None
    validate_at_end: bool = True

    def experiment(self):
        operational = {"output_root", "resume", "stop_after", "max_seconds", "checkpoint_every",
                       "checkpoint_seconds", "validate_at_end"}
        return {k: v for k, v in asdict(self).items() if k not in operational}

    def check(self):
        if not self.run_id or self.run_id in (".", "..") or any(c in self.run_id for c in "/\\:"):
            raise ValueError("run_id must be one safe directory name")
        for name in ("steps", "batch_size", "threads", "checkpoint_every", "candidate_chunk_size", "max_padded_candidates"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.validation_every < 0 or self.checkpoint_seconds <= 0 or self.temperature <= 0:
            raise ValueError("Invalid validation/checkpoint interval or temperature")
        if self.lr <= 0 or self.weight_decay < 0 or min(self.alignment_weight, self.transition_weight) < 0:
            raise ValueError("Invalid optimizer or loss coefficients")
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive or null")
        if self.stop_after is not None and self.stop_after < 1:
            raise ValueError("stop_after must be positive")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Select a GPU runtime in Colab, then rerun preflight.")
        if self.amp and self.device != "cuda":
            raise ValueError("AMP requires CUDA")


def load_config(path, **overrides):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    value.update({k: v for k, v in overrides.items() if v is not None})
    return TrainConfig(**value)


def software_provenance():
    package = Path(__file__).resolve().parent
    core_hash = object_hash({str(p.relative_to(package)).replace("\\", "/"):
                            hashlib.sha256(p.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
                            for p in sorted(package.rglob("*.py"))})
    result = subprocess.run(["git", "-C", str(package.parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return {"package": "unisar-pytorch", "implementation_hash": core_hash,
            "source_revision": result.stdout.strip() if result.returncode == 0 else None,
            "torch_version": str(torch.__version__), "python": platform.python_version()}


def experiment_hash(config):
    return object_hash({"experiment": config.experiment(),
                        "implementation_hash": software_provenance()["implementation_hash"]})


class TaskSampler:
    """Lazy search-group uniform / rec-exposure uniform sampling with replacement."""
    def __init__(self, search, recommendation, seed):
        if not len(search) or not len(recommendation):
            raise ValueError("Both training tasks are required")
        self.search, self.recommendation = search, recommendation
        self.rng, self.cursor = random.Random(seed), 0
        self.rec_offsets = []
        total = 0
        # Reading raw cardinalities avoids expanding every encoded history at startup.
        for i in range(len(recommendation)):
            if hasattr(recommendation, "indexes"):
                n = len(recommendation.dataset.raw_requests[recommendation.indexes[i]]["item_raw"])
            else:
                n = len(recommendation[i]["item_ids"])
            if n < 1:
                raise ValueError("Empty recommendation group")
            total += n
            self.rec_offsets.append(total)
        self.rec_count = total

    def batch(self, task, size):
        if task == "search":
            result = [self.search[self.rng.randrange(len(self.search))] for _ in range(size)]
        elif task == "recommendation":
            result = []
            for _ in range(size):
                offset = self.rng.randrange(self.rec_count)
                row = bisect.bisect_right(self.rec_offsets, offset)
                column = offset - (self.rec_offsets[row - 1] if row else 0)
                example = dict(self.recommendation[row])
                for key in ("item_ids", "caption_ids", "labels", "item_raw", "metadata_present"):
                    if key in example:
                        example[key] = [example[key][column]]
                result.append(example)
        else:
            raise ValueError("Unknown task")
        self.cursor += size
        return result

    def state_dict(self):
        return {"rng": self.rng.getstate(), "cursor": self.cursor}

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
        self.cursor = int(state["cursor"])


def microbatches(examples, budget):
    """Preserve whole candidate groups; cap padding waste between groups."""
    current, longest = [], 0
    for example in sorted(examples, key=lambda x: len(x["item_ids"])):
        size = len(example["item_ids"])
        if current and (len(current) + 1) * max(longest, size) > budget:
            yield current
            current, longest = [], 0
        current.append(example)
        longest = max(longest, size)
    if current:
        yield current


def score_requests(model, requests, device="cpu", batch_size=32, candidate_chunk_size=256,
                   max_padded_candidates=4096):
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(requests), batch_size):
            for part in microbatches(requests[start:start + batch_size], max_padded_candidates):
                batch = to_device(collate_requests(part, model.max_history), device)
                scores = model(batch, candidate_chunk_size=candidate_chunk_size).float().cpu().tolist()
                for request, values in zip(part, scores):
                    common = {"task": request["task"], "user_raw": request["user_raw"],
                              "request_id": request["request_id"], "cohort": request.get("cohort", "ambiguous")}
                    values = values[:len(request["item_ids"])]
                    if request["task"] == "search":
                        yield {**common, "item_raw": request["item_raw"], "labels": request["labels"],
                               "scores": values, "fallback": False}
                    else:
                        for item, label, value in zip(request["item_raw"], request["labels"], values):
                            yield {**common, "item_raw": item, "label": label, "score": value, "fallback": False}


def validate(model, requests, config):
    rows = list(score_requests(model, requests, config.device, config.batch_size,
                              config.candidate_chunk_size, config.max_padded_candidates))
    search = [r for r in rows if r["task"] == "search"]
    rec = [r for r in rows if r["task"] == "recommendation"]
    if not search or not rec:
        raise ValueError("Validation requires both tasks")
    def bce(labels, logits):
        y, z = np.asarray(labels, dtype=np.float64), np.asarray(logits, dtype=np.float64)
        return float(np.mean(np.logaddexp(0, z) - y * z))
    search_bce = float(np.mean([bce(r["labels"], r["scores"]) for r in search]))
    rec_bce = bce([r["label"] for r in rec], [r["score"] for r in rec])
    return {"search_group_log_loss": search_bce, "recommendation_log_loss": rec_bce,
            "joint_log_loss": .5 * (search_bce + rec_bce), "search": evaluate_search(search),
            "recommendation": evaluate_recommendation([r["label"] for r in rec], [r["score"] for r in rec])}, rows


def write_predictions(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def train(dataset, config: TrainConfig):
    """The only entry point that starts optimization; final test is never loaded."""
    config.check()
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(config.device == "cpu")
    architecture = ModelConfig(item_count=dataset.item_count, token_count=dataset.token_count, **config.model)
    model = UniSAR(architecture).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    warmup = max(1, math.ceil(config.steps * .05))
    def schedule(n):
        if n < warmup:
            return (n + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(1., (n - warmup) / max(1, config.steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    sampler = TaskSampler(dataset.requests("train", "search"), dataset.requests("train", "recommendation"), config.seed + 1)
    run_dir = Path(config.output_root) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = run_dir / "latest.pt"
    software = software_provenance()
    config_hash, data_hash = experiment_hash(config), str(dataset.data_hash)
    step, best, records, seen_search, seen_rec, seen_candidates, cumulative_seconds = 0, None, [], 0, 0, 0, 0.
    if config.resume:
        snapshot = load_checkpoint(latest if config.resume == "latest" else config.resume,
            models=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            config_hash=config_hash, data_hash=data_hash, map_location=config.device,
            expected_run_id=config.run_id)
        step = snapshot["step"]
        sampler.load_state_dict(snapshot["sampler"])
        extra = snapshot["extra"]
        best, records = extra["best"], extra["records"]
        seen_search, seen_rec, seen_candidates = extra["seen_search"], extra["seen_rec"], extra["seen_candidates"]
        cumulative_seconds = extra["training_seconds"]
    elif latest.exists() or (run_dir / "config.json").exists():
        raise FileExistsError("Run exists. Resume it or use a new run_id.")
    else:
        write_json(run_dir / "config.json", {**asdict(config), "architecture": asdict(architecture),
                   "config_hash": config_hash, "data_hash": data_hash,
                   "software": software,
                   "selection": "minimum full-validation equal-task supervised BCE",
                   "protocol": "clean identities-only block history; logged candidates; train-only vocabulary"})
    call_start, last_save = time.perf_counter(), time.perf_counter()
    initial_step, training_seconds = step, 0.
    peak_rss = psutil.Process().memory_info().rss
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    def save(path):
        save_checkpoint(path, models=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            step=step, sampler_state=sampler.state_dict(), config_hash=config_hash, data_hash=data_hash,
            config=asdict(config), extra={"best": best, "records": records, "seen_search": seen_search,
                "seen_rec": seen_rec, "seen_candidates": seen_candidates,
                "training_seconds": cumulative_seconds + training_seconds, "architecture": asdict(architecture),
                "software": software})
    if not config.resume:
        save(latest)
    def validation():
        nonlocal best
        metrics, rows = validate(model, dataset.requests("validation"), config)
        records.append({"step": step, "validation": metrics})
        if best is None or metrics["joint_log_loss"] < best:
            best = metrics["joint_log_loss"]
            save(run_dir / "best.pt")
            write_json(run_dir / "validation-best.json", {"step": step, **metrics})
            write_predictions(run_dir / "validation-best.jsonl", rows)
        return metrics
    target = min(config.steps, config.stop_after) if config.stop_after else config.steps
    last_validated = -1
    status = "complete"
    safe = True
    try:
        while step < target:
            if config.max_seconds is not None and time.perf_counter() - call_start >= config.max_seconds:
                status = "time_budget_reached"
                break
            safe = False
            tick = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses_by_task, candidate_count = [], 0
            for task in ("search", "recommendation"):
                examples = sampler.batch(task, config.batch_size)
                if task == "search":
                    candidate_count = sum(len(r["item_ids"]) for r in examples)
                pieces = []
                for part in microbatches(examples, config.max_padded_candidates):
                    batch = to_device(collate_requests(part, model.max_history), config.device)
                    with torch.autocast(device_type=config.device, enabled=config.amp):
                        output = model.forward_with_aux(batch, candidate_chunk_size=config.candidate_chunk_size)
                        losses = joint_loss(output, batch, alignment_weight=config.alignment_weight,
                                            transition_weight=config.transition_weight, temperature=config.temperature)
                    loss = losses["total"]
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite {task} loss at step {step}")
                    fraction = len(part) / len(examples)
                    scaler.scale(.5 * fraction * loss).backward()
                    pieces.append({k: fraction * float(v.detach().cpu()) for k, v in losses.items()
                                   if isinstance(v, torch.Tensor) and v.numel() == 1})
                losses_by_task.append({k: sum(x.get(k, 0.) for x in pieces) for k in pieces[0]})
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite gradient at step {step}")
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Do not advance schedule/count for an AMP overflow-skipped update.
            if scaler.get_scale() < previous_scale:
                raise FloatingPointError("AMP overflow skipped update. Diagnose before resuming the same configuration, or start a new FP32 run; edited AMP settings are not an exact resume.")
            scheduler.step()
            if config.device == "cuda":
                torch.cuda.synchronize()
            step += 1
            seen_search += config.batch_size
            seen_rec += config.batch_size
            seen_candidates += candidate_count
            duration = time.perf_counter() - tick
            training_seconds += duration
            safe = True
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
            if step == 1 or step % 50 == 0 or step == target:
                record = {"step": step, "search": losses_by_task[0], "recommendation": losses_by_task[1],
                          "step_seconds": duration, "lr": optimizer.param_groups[0]["lr"]}
                records.append(record)
                print(json.dumps(record), flush=True)
            if step % config.checkpoint_every == 0 or time.perf_counter() - last_save >= config.checkpoint_seconds:
                save(latest)
                last_save = time.perf_counter()
            if config.validation_every and step % config.validation_every == 0:
                save(latest)  # Persist before potentially expensive full validation.
                validation()
                last_validated = step
        if step < config.steps and status == "complete":
            status = "bounded_check_complete"
        save(latest)
        if config.validate_at_end and last_validated != step:
            validation()
        save(latest)
    except KeyboardInterrupt:
        if safe:
            save(latest)
        raise
    except BaseException:
        # Keep the preceding atomic snapshot if an update was interrupted.
        if safe:
            save(latest)
        raise
    report = {"status": status, "completed_steps": step, "planned_steps": config.steps,
              "data_hash": data_hash, "config_hash": config_hash,
              "software": software,
              "parameters": sum(p.numel() for p in model.parameters()),
              "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
              "peak_rss_bytes_observed": peak_rss,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if config.device == "cuda" else None,
              "call_wall_seconds": time.perf_counter() - call_start,
              "training_seconds": cumulative_seconds + training_seconds,
              "mean_call_step_seconds": training_seconds / (step - initial_step) if step > initial_step else None,
              "search_group_presentations": seen_search, "recommendation_exposure_presentations": seen_rec,
              "search_candidate_presentations": seen_candidates, "best_validation_joint_log_loss": best,
              "hardware": {"cpu": platform.processor(), "torch": str(torch.__version__),
                           "device": config.device, "threads": config.threads,
                           "gpu": torch.cuda.get_device_name() if config.device == "cuda" else None},
              "records": records, "note": "Time cap checked between updates; validation and checkpoint I/O can extend wall time."}
    write_json(run_dir / "metrics.json", report)
    return report


def export_checkpoint(checkpoint, output):
    """Trusted local training snapshot -> weights-only portable inference payload."""
    snapshot = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if snapshot.get("format_version") != 1:
        raise ValueError("Unsupported training snapshot")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "unisar-weights-v1", "architecture": snapshot["extra"]["architecture"],
               "state_dict": {k: v.cpu() for k, v in snapshot["models"].items()},
               "data_hash": snapshot["data_hash"], "config_hash": snapshot["config_hash"]}
    temp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(target)
    return {"path": str(target), "bytes": target.stat().st_size, "sha256": digest(target)}


def load_export(path, data_hash=None, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "unisar-weights-v1":
        raise ValueError("Unsupported weights format")
    if data_hash is not None and payload["data_hash"] != data_hash:
        raise ValueError("Export/data contract hash mismatch")
    model = UniSAR(ModelConfig(**payload["architecture"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--data", required=True)
    fit.add_argument("--config", required=True)
    fit.add_argument("--output-root")
    fit.add_argument("--run-id")
    fit.add_argument("--device", choices=("cpu", "cuda"))
    fit.add_argument("--max-seconds", type=float)
    fit.add_argument("--stop-after", type=int)
    fit.add_argument("--resume", nargs="?", const="latest")
    fit.add_argument("--no-final-validation", action="store_true")
    exp = sub.add_parser("export")
    exp.add_argument("--checkpoint", required=True)
    exp.add_argument("--output", required=True)
    val = sub.add_parser("validate")
    val.add_argument("--data", required=True)
    val.add_argument("--weights", required=True)
    val.add_argument("--output", required=True)
    val.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.command == "export":
        print(json.dumps(export_checkpoint(args.checkpoint, args.output)))
        return
    from unisar.data.dataset import PreparedDataset
    ds = PreparedDataset(args.data)
    if args.command == "train":
        config = load_config(args.config, output_root=args.output_root, run_id=args.run_id,
                             device=args.device, max_seconds=args.max_seconds, stop_after=args.stop_after,
                             resume=args.resume, validate_at_end=False if args.no_final_validation else None)
        print(json.dumps({k: v for k, v in train(ds, config).items() if k != "records"}))
    else:
        model = load_export(args.weights, ds.data_hash, args.device)
        config = TrainConfig(device=args.device)
        metrics, rows = validate(model, ds.requests("validation"), config)
        directory = Path(args.output)
        write_json(directory / "validation.json", metrics)
        write_predictions(directory / "validation.jsonl", rows)
        print(json.dumps({"joint_log_loss": metrics["joint_log_loss"], "output": str(directory)}))


if __name__ == "__main__":
    main()
