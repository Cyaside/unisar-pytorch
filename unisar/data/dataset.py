"""Lazy encoded request views; prefix pointers avoid expanded history storage."""
from __future__ import annotations
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
import pyarrow.parquet as pq
from unisar.common import digest,read_json

class RequestView(Sequence):
    def __init__(self,dataset,indexes):self.dataset,self.indexes=dataset,indexes
    def __len__(self):return len(self.indexes)
    def __getitem__(self,index):
        if isinstance(index,slice):return RequestView(self.dataset,self.indexes[index])
        return self.dataset.encode(self.indexes[index])

class PreparedDataset:
    def __init__(self,root:Path|str,verify=True):
        self.root=Path(root)
        self.manifest=read_json(self.root/"manifest.json")
        if verify:
            for name,expected in self.manifest["files"].items():
                if digest(self.root/name)!=expected:raise ValueError(f"Prepared hash mismatch: {name}")
        self.data_hash=self.manifest["data_hash"]
        self.item_count=self.manifest["item_count"]
        self.token_count=self.manifest["token_count"]
        self.vocab=read_json(self.root/"vocabulary.json")
        self.item_map={raw:i+2 for i,raw in enumerate(self.vocab["items"])}
        self.token_map={raw:i+2 for i,raw in enumerate(self.vocab["tokens"])}
        self.raw_requests=pq.read_table(self.root/"requests.parquet").to_pylist()
        self.blocks=pq.read_table(self.root/"blocks.parquet").to_pylist()
        self.metadata={m["item_raw"]:m for m in pq.read_table(self.root/"metadata.parquet").to_pylist()}
    def token_ids(self,raw):return [self.token_map.get(t,1) for t in raw[:50]]
    @lru_cache(maxsize=50000)
    def encode_block(self,index):
        b=self.blocks[index]
        return {"timestamp_ms":b["timestamp_ms"],"rec_item_ids":[self.item_map.get(i,1) for i in b["rec_items"]],
            "search_query_ids":[self.token_ids(q) for q in b["search_queries"]],"service_mask":b["service_mask"],"gap_bucket":b["gap_bucket"]}
    def encode(self,index):
        r=self.raw_requests[index]
        return {**r,"item_ids":[self.item_map.get(i,1) for i in r["item_raw"]],
            "caption_ids":[self.token_ids(self.metadata.get(i,{}).get("caption_tokens",[])) for i in r["item_raw"]],
            "metadata_present":[i in self.metadata for i in r["item_raw"]],"query_ids":self.token_ids(r["query_tokens"]),
            "history":[self.encode_block(i) for i in r["history_ids"]]}
    def requests(self,split,task=None):
        if split not in ("train","validation","test"):raise ValueError("Unknown split")
        return RequestView(self,[i for i,r in enumerate(self.raw_requests) if r["split"]==split and (task is None or r["task"]==task)])
