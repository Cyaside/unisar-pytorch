"""Pure chronology contracts shared by preparation and regression fixtures."""
from __future__ import annotations
from bisect import bisect_left
from unisar.data.audit import CUTOFFS

GAP_THRESHOLDS = (60000,300000,1800000,7200000,86400000)
FEATURE_POLICY = {
    "version":"identities-blocks-v1", "history_limit":50,"token_cap":50,
    "history":"prior recommendation distribution IDs and issued search query IDs",
    "strict_prefix":True,"simultaneous":"pool unordered; score before append",
    "feedback":False,"search_result_history":False,"user_attributes":False,
    "metadata":"static released snapshot; ID-only sensitivity required",
    "gap_thresholds_ms":GAP_THRESHOLDS,"transition_window_ms":1800000,
    "pad":0,"oov":1,
}

def split_at(timestamp: int) -> str:
    return ("train","validation","test","unused")[bisect_left(CUTOFFS, timestamp+1)]

def gap_bucket(gap: int | None) -> int:
    if gap is None:
        return 0
    if gap <= 0:
        raise ValueError("History must be strictly earlier")
    return bisect_left(GAP_THRESHOLDS, gap)+1

def block_summary(rec_items=(), search_queries=()) -> dict:
    rec = sorted(set(rec_items))
    queries = [list(q) for q in sorted(set(tuple(q) for q in search_queries))]
    return {"rec_items":rec,"search_queries":queries,"service_mask":(2 if rec else 0)|(1 if queries else 0)}

def cohort_for(previous: dict | None, timestamp: int, task: str) -> str:
    if previous is None or previous["service_mask"] not in (1,2):
        return "ambiguous"
    gap = timestamp-previous["timestamp_ms"]
    if gap <= 0:
        raise ValueError("Previous block is not earlier")
    if gap > FEATURE_POLICY["transition_window_ms"]:
        return "ambiguous"
    return "same_service" if previous["service_mask"] == (1 if task=="search" else 2) else "transition"

def attach_prefixes(blocks: list[dict], requests: list[dict], max_history=50) -> list[dict]:
    """Annotate requests without consulting labels; immutable future append safety."""
    indexes: dict[int,list[int]] = {}
    current: dict[tuple[int,int],list[dict]] = {}
    for request in requests:
        current.setdefault((request["user_raw"],request["timestamp_ms"]),[]).append(request)
    out=[]
    for idx,block in enumerate(blocks):
        history=indexes.setdefault(block["user_raw"],[])
        previous=blocks[history[-1]] if history else None
        if previous and previous["timestamp_ms"] >= block["timestamp_ms"]:
            raise ValueError("Blocks must be unique and chronological per user")
        for request in current.get((block["user_raw"],block["timestamp_ms"]),[]):
            out.append({**request,"history_ids":history[-max_history:].copy(),"history_length_total":len(history),
                "request_gap":gap_bucket(block["timestamp_ms"]-previous["timestamp_ms"] if previous else None),
                "cohort":cohort_for(previous,block["timestamp_ms"],request["task"])})
        history.append(idx)
    if len(out)!=len(requests):
        raise ValueError("Request lacks a matching block")
    return out
