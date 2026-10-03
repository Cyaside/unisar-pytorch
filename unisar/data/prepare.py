"""Build immutable request/prefix pointers and train-only compact vocabularies."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from unisar.common import DATA_ROOT,ROOT,digest,object_hash,read_json,reports_dir,write_json
from unisar.data.audit import tokens
from unisar.data.temporal import FEATURE_POLICY,attach_prefixes,block_summary,gap_bucket,split_at

def build(canonical: Path, output: Path) -> dict:
    if (output / "manifest.json").exists():
        raise FileExistsError("Prepared manifest exists; reuse it or choose a new output root to preserve its history.")
    output.mkdir(parents=True,exist_ok=True)
    source=read_json(canonical/"manifest.json")
    for name,expected in source["files"].items():
        if digest(canonical/name)!=expected:
            raise ValueError(f"Canonical hash mismatch: {name}")
    search=pq.read_table(canonical/"search.parquet").to_pylist()
    rec=pq.read_table(canonical/"recommendation.parquet").to_pylist()
    metadata=pq.read_table(canonical/"metadata.parquet").to_pylist()
    components=defaultdict(lambda:{"rec_items":[],"search_queries":[]})
    requests=[]
    train_items=set()
    train_tokens=set()
    for row in search:
        q=tokens(row["query_signature"])
        key=(row["user_raw"],row["timestamp_ms"])
        components[key]["search_queries"].append(q)
        split=split_at(row["timestamp_ms"])
        if split=="train":
            train_tokens.update(q)
            train_items.update(row["item_raw"])
        if row["item_raw"] and split!="unused":
            requests.append({"user_raw":row["user_raw"],"request_id":f"s:{row['user_raw']}:{row['session_raw']}",
                "timestamp_ms":row["timestamp_ms"],"task":"search","item_raw":row["item_raw"],
                "labels":row["labels"],"query_tokens":q,"split":split})
    grouped_rec=defaultdict(list)
    for row in rec:
        key=(row["user_raw"],row["timestamp_ms"])
        components[key]["rec_items"].append(row["item_raw"])
        grouped_rec[key].append(row)
        if split_at(row["timestamp_ms"])=="train":
            train_items.add(row["item_raw"])
    for (user,ts),rows in sorted(grouped_rec.items()):
        split=split_at(ts)
        if split!="unused":
            requests.append({"user_raw":user,"request_id":f"r:{user}:{ts}","timestamp_ms":ts,"task":"recommendation",
                "item_raw":[r["item_raw"] for r in rows],"labels":[r["label"] for r in rows],"query_tokens":[],"split":split})
    needed_items={i for v in components.values() for i in v["rec_items"]}|{i for r in search for i in r["item_raw"]}
    metadata=sorted((m for m in metadata if m["item_raw"] in needed_items),key=lambda m:m['item_raw'])
    meta=[]
    for row in metadata:
        caption=tokens(row.pop("caption"))
        row["caption_tokens"]=caption
        meta.append(row)
        if row["item_raw"] in train_items:
            train_tokens.update(caption)
    blocks=[]
    prev_by_user={}
    for (user,ts),values in sorted(components.items()):
        prev=prev_by_user.get(user)
        blocks.append({"user_raw":user,"timestamp_ms":ts,**block_summary(**values),"gap_bucket":gap_bucket(ts-prev if prev is not None else None)})
        prev_by_user[user]=ts
    requests=attach_prefixes(blocks,requests,FEATURE_POLICY["history_limit"])
    requests.sort(key=lambda r:(r["user_raw"],r["timestamp_ms"],r["request_id"]))
    vocab={"items":sorted(train_items),"tokens":sorted(train_tokens),"pad":0,"oov":1,"fit":"training exposures/issued queries; metadata of training-seen items only"}
    write_json(output/"vocabulary.json",vocab)
    write_json(output/"feature-policy.json",FEATURE_POLICY)
    pq.write_table(pa.Table.from_pylist(blocks),output/"blocks.parquet",compression="zstd")
    pq.write_table(pa.Table.from_pylist(requests),output/"requests.parquet",compression="zstd")
    pq.write_table(pa.Table.from_pylist(meta),output/"metadata.parquet",compression="zstd")
    counts={}
    metadata_ids={m['item_raw'] for m in meta}
    for split in ("train","validation","test"):
        selected=[r for r in requests if r["split"]==split]
        counts[split]={"requests_by_task":dict(Counter(r["task"] for r in selected)),
            "users":len({r["user_raw"] for r in selected}),"search_cohorts":dict(Counter(r["cohort"] for r in selected if r["task"]=="search")),
            "exposures":sum(len(r["item_raw"]) for r in selected),"oov_items":sum(i not in train_items for r in selected for i in r["item_raw"]),
            "missing_metadata_exposures":sum(i not in metadata_ids for r in selected for i in r["item_raw"])}
    files={name:digest(output/name) for name in ("requests.parquet","blocks.parquet","metadata.parquet","vocabulary.json","feature-policy.json")}
    coverage={"query_lists_over_token_cap":sum(len(r["query_tokens"])>50 for r in requests),"query_lists":len(requests),
        "caption_lists_over_token_cap":sum(len(m["caption_tokens"])>50 for m in meta),"caption_lists":len(meta),
        "history_truncated_requests":sum(r["history_length_total"]>50 for r in requests)}
    manifest={"canonical_hash":source["data_hash"],"raw_archive_sha256":source.get("raw_archive_sha256"),"cohort_hash":source["cohort_hash"],"schema_version":"prepared-blocks-v1",
        "feature_policy_hash":object_hash(FEATURE_POLICY),"vocabulary_hash":object_hash(vocab),"split_hash":object_hash({"cutoffs":source['cutoffs_utc8_ms'],"counts":counts}),
        "cutoffs_utc8_ms":source["cutoffs_utc8_ms"],"counts":counts,"coverage":coverage,"files":files,
        "item_count":len(train_items)+2,"token_count":len(train_tokens)+2,"blocks":len(blocks)}
    manifest["data_hash"]=object_hash(manifest)
    write_json(output/"manifest.json",manifest)
    report=reports_dir(ROOT)/f"temporal-{output.name}.md"
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text("# Temporal preparation: "+output.name+"\n\nTraining-only vocabulary; strictly earlier, unordered block history; no historical outcomes or search result IDs. Static catalog assumption applies to current candidate captions.\n\n```json\n"+__import__('json').dumps(manifest,indent=2)+"\n```\n",encoding="utf-8")
    print(f"prepared {output}: {len(requests)} requests, {len(blocks)} blocks, {manifest['item_count']} item IDs",flush=True)
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--canonical",type=Path,default=DATA_ROOT/"canonical/pilot")
    p.add_argument("--output",type=Path,default=DATA_ROOT/"prepared/pilot-v1")
    a=p.parse_args();build(a.canonical,a.output)
