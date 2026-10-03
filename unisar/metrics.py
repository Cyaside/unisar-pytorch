"""TRACE's logged-candidate evaluator; no parameters are fitted here.

Scores are finite, uncalibrated logits. Search ties use ascending numeric raw
item IDs. Primary ranking excludes singleton, no-positive, and all-positive
requests, in that mutually exclusive order. All requests remain in log loss.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

EVALUATOR_VERSION = "trace-logged-v1"
COHORTS = ("transition", "same_service", "ambiguous")
METRICS = ("ndcg", "recall", "mrr")


def _number(value: object, name: str) -> float:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be numeric, not text")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _binary(value: object) -> int:
    result = _number(value, "label")
    if result not in (0.0, 1.0):
        raise ValueError("labels must be binary 0/1")
    return int(result)


def _raw_item(value: object) -> int:
    # Float IDs can silently round 64-bit identifiers before evaluation.
    if isinstance(value, bool):
        raise ValueError("item_raw must contain integer IDs, not booleans")
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if isinstance(value, int):
        return value
    # Support NumPy integer scalars without accepting floating-point IDs.
    if hasattr(value, "__index__"):
        return int(value.__index__())
    raise ValueError("item_raw must contain integers or integral-ID strings")


def _identity(value: object, name: str) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a nonempty raw identifier")
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be a nonempty raw identifier")
    return result


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(value / len(values) for value in values) if values else None


def _log_loss(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    losses = [max(z, 0.0) - z * y + math.log1p(math.exp(-abs(z)))
              for y, z in zip(labels, scores)]
    return _mean(losses)


def search_request_metrics(record: Mapping, k: int = 10) -> dict:
    """Validate one request and retain its metrics for paired cluster analysis.

    Relevance labels never resolve score ties. Each result includes its user,
    request ID, cohort, and mutually exclusive ranking exclusion reason.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    required = {"user_raw", "request_id", "item_raw", "labels", "scores", "cohort"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"missing search fields: {sorted(missing)}")
    user = _identity(record["user_raw"], "user_raw")
    request = _identity(record["request_id"], "request_id")
    cohort = record["cohort"]
    if cohort not in COHORTS:
        raise ValueError(f"cohort must be one of {COHORTS}; received {cohort!r}")
    try:
        items = [_raw_item(value) for value in record["item_raw"]]
        labels = [_binary(value) for value in record["labels"]]
        scores = [_number(value, "score") for value in record["scores"]]
    except TypeError as exc:
        raise ValueError("item_raw, labels and scores must be sequences") from exc
    if not items or len(items) != len(labels) or len(items) != len(scores):
        raise ValueError("search candidate/label/score lengths must match and be nonzero")
    if len(set(items)) != len(items):
        raise ValueError("duplicate raw candidate IDs; deduplicate upstream before evaluation")
    positives = sum(labels)
    exclusion = ("singleton" if len(items) == 1 else
                 "no_positive" if positives == 0 else
                 "all_positive" if positives == len(items) else None)
    result = {"user_raw": user, "request_id": request, "cohort": cohort,
              "candidates": len(items), "positives": positives,
              "exclusion": exclusion, "log_loss": _log_loss(labels, scores),
              **{metric: None for metric in METRICS}}
    if exclusion is None:
        order = sorted(range(len(items)), key=lambda index: (-scores[index], items[index]))[:k]
        ranked_labels = [labels[index] for index in order]
        dcg = math.fsum(label / math.log2(rank + 2) for rank, label in enumerate(ranked_labels))
        ideal = math.fsum(1.0 / math.log2(rank + 2) for rank in range(min(k, positives)))
        first_click = next((rank + 1 for rank, label in enumerate(ranked_labels) if label), None)
        result.update(ndcg=dcg / ideal, recall=sum(ranked_labels) / positives,
                      mrr=1.0 / first_click if first_click else 0.0)
    return result


def _aggregate_search(rows: Sequence[dict], k: int) -> dict:
    by_user = defaultdict(list)
    exclusions = {name: 0 for name in ("singleton", "no_positive", "all_positive")}
    for row in rows:
        if row["exclusion"]:
            exclusions[row["exclusion"]] += 1
        else:
            by_user[row["user_raw"]].append(row)
    eligible = [row for user_rows in by_user.values() for row in user_rows]
    all_users = {row["user_raw"] for row in rows}
    total_candidates = sum(row["candidates"] for row in rows)
    return {
        "total_requests": len(rows), "total_users": len(all_users),
        "eligible_requests": len(eligible), "eligible_users": len(by_user),
        "users_without_eligible_request": len(all_users) - len(by_user),
        "total_candidates": total_candidates,
        "positive_candidates": sum(row["positives"] for row in rows),
        "exclusions": exclusions,
        "exclusion_fractions": {name: count / len(rows) if rows else None
                                for name, count in exclusions.items()},
        "request_macro": {f"{metric}@{k}": _mean([row[metric] for row in eligible])
                          for metric in METRICS},
        "user_macro": {
            f"{metric}@{k}": _mean([_mean([row[metric] for row in user_rows])
                                     for user_rows in by_user.values()])
            for metric in METRICS},
        # Candidate-weighted BCE includes every observed label, including
        # requests excluded from the primary ranking denominator.
        "log_loss": math.fsum(row["log_loss"] * (row["candidates"] / total_candidates)
                              for row in rows) if rows else None,
    }


def evaluate_search(records: Iterable[Mapping], k: int = 10) -> dict:
    """Search metrics overall and in each fixed service cohort.

    Empty eligible denominators produce JSON null, never a fabricated zero.
    Request IDs are unique within a user; duplicate prediction rows fail.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    rows = []
    seen = set()
    for index, record in enumerate(records):
        try:
            row = search_request_metrics(record, k)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"search record {index}: {exc}") from exc
        identity = (row["user_raw"], row["request_id"])
        if identity in seen:
            raise ValueError(f"duplicate search request {identity!r}")
        seen.add(identity)
        rows.append(row)
    return {"evaluator_version": EVALUATOR_VERSION, "k": k,
            "overall": _aggregate_search(rows, k),
            **{cohort: _aggregate_search([row for row in rows if row["cohort"] == cohort], k)
               for cohort in COHORTS}}


def evaluate_recommendation(labels: Iterable, scores: Iterable) -> dict:
    """Exposure log loss, noninterpolated AP, and ROC-AUC from finite logits.

    AP handles tied scores as one threshold, matching its standard definition.
    Empty inputs yield null metrics. AP is null with no positives; ROC-AUC is
    null with a constant-label population. No sigmoid saturation affects ranks.
    """
    labels = [_binary(value) for value in labels]
    scores = [_number(value, "score") for value in scores]
    if len(labels) != len(scores):
        raise ValueError("recommendation label/score lengths must match")
    count = len(labels)
    positives = sum(labels)
    negatives = count - positives
    groups = defaultdict(lambda: [0, 0])
    for label, score in zip(labels, scores):
        groups[score][label] += 1
    tp = fp = 0
    ap_terms = []
    auc_terms = []
    for score in sorted(groups, reverse=True):
        group_neg, group_pos = groups[score]
        tp += group_pos
        fp += group_neg
        if positives:
            ap_terms.append((group_pos / positives) * (tp / (tp + fp)))
        if positives and negatives:
            neg_below = negatives - fp
            auc_terms.append((group_pos / positives) * ((neg_below + 0.5 * group_neg) / negatives))
    return {"evaluator_version": EVALUATOR_VERSION,
            "count": count, "positives": positives, "negatives": negatives,
            "prevalence": positives / count if count else None,
            "log_loss": _log_loss(labels, scores),
            "average_precision": math.fsum(ap_terms) if positives else None,
            "roc_auc": math.fsum(auc_terms) if positives and negatives else None,
            "undefined": {"average_precision": "no_positive" if not positives else None,
                          "roc_auc": "not_two_classes" if not positives or not negatives else None}}


def evaluate_predictions(records: Iterable[Mapping], k: int = 10) -> dict:
    """Evaluate mixed JSONL: search requests or singular recommendation rows."""
    search = []
    rec_labels, rec_scores = [], []
    for index, record in enumerate(records):
        task = record.get("task", "search")
        if task == "search":
            search.append(record)
        elif task in ("recommendation", "rec"):
            if "label" not in record or "score" not in record:
                raise ValueError(f"recommendation record {index} needs label and score")
            rec_labels.append(record["label"])
            rec_scores.append(record["score"])
        else:
            raise ValueError(f"record {index}: unsupported task {task!r}")
    return {"search": evaluate_search(search, k),
            "recommendation": evaluate_recommendation(rec_labels, rec_scores)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Logged prediction JSONL")
    parser.add_argument("--output", type=Path, help="JSON aggregate report; otherwise stdout")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    records = []
    with args.predictions.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number}: expected a JSON object")
                records.append(value)
    report = json.dumps(evaluate_predictions(records, args.k), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
