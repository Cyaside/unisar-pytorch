from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from unisar.common import digest,write_json
from unisar.data.audit import CUTOFFS
from unisar.data.prepare import build
from unisar.data.dataset import PreparedDataset

def test_train_only_vocabulary_and_reproducible_prep(tmp_path,monkeypatch):
    import unisar.data.prepare as preparation
    monkeypatch.setattr(preparation,'ROOT',tmp_path)
    canonical=tmp_path/'canonical';canonical.mkdir()
    t=CUTOFFS[0]
    search=[{'user_raw':1,'session_raw':1,'timestamp_ms':t-1000,'query_signature':'[3]', 'item_raw':[7,8], 'labels':[1,0],'raw_count':2,'retained_raw_count':2},
        {'user_raw':1,'session_raw':2,'timestamp_ms':t+1000,'query_signature':'[99]','item_raw':[9,7],'labels':[0,1],'raw_count':2,'retained_raw_count':2}]
    rec=[{'user_raw':1,'item_raw':7,'timestamp_ms':t-2000,'label':1,'raw_row':1},
        {'user_raw':1,'item_raw':8,'timestamp_ms':t-2000,'label':0,'raw_row':2}]
    metadata=[{'item_raw':7,'caption':'[11, 12]','author_raw':1,'category_raw':1,'category_name':'test'},
        {'item_raw':8,'caption':'[13]','author_raw':1,'category_raw':1,'category_name':'test'},
        {'item_raw':9,'caption':'[77]','author_raw':1,'category_raw':1,'category_name':'test'}]
    for name,rows in [('search',search),('recommendation',rec),('metadata',metadata)]:
        pq.write_table(pa.Table.from_pylist(rows),canonical/f'{name}.parquet')
    write_json(canonical/'users.json',[1])
    write_json(canonical/'manifest.json',{'data_hash':'fixture','cohort_hash':'fixture','cutoffs_utc8_ms':CUTOFFS,
        'files':{name:digest(canonical/name) for name in ('users.json','search.parquet','recommendation.parquet','metadata.parquet')}})
    first=build(canonical,tmp_path/'first');second=build(canonical,tmp_path/'second')
    assert first['data_hash']==second['data_hash']
    ds=PreparedDataset(tmp_path/'first')
    assert 9 not in ds.item_map and 77 not in ds.token_map and 99 not in ds.token_map
    val=ds.requests('validation','search')[0]
    assert val['item_ids'][0]==1 and val['caption_ids'][0]==[1] and val['query_ids']==[1]
    assert [b['timestamp_ms'] for b in val['history']]==[t-2000,t-1000]
    assert val['history'][1]['rec_item_ids']==[] # search result item IDs excluded
    assert all(b['timestamp_ms']<val['timestamp_ms'] for b in val['history'])


import copy
import pytest
from unisar.data.temporal import attach_prefixes,block_summary,split_at,gap_bucket,cohort_for
from unisar.data.audit import CUTOFFS,tokens

def fixture():
    blocks=[{"user_raw":1,"timestamp_ms":100,**block_summary([9,7,9],[])},
        {"user_raw":1,"timestamp_ms":200,**block_summary([5],[[3,4],[2]])},
        {"user_raw":1,"timestamp_ms":300,**block_summary([],[[6]])}]
    requests=[{"user_raw":1,"timestamp_ms":200,"task":"search","labels":[1,0]},
        {"user_raw":1,"timestamp_ms":200,"task":"recommendation","labels":[0]},
        {"user_raw":1,"timestamp_ms":300,"task":"search","labels":[1]}]
    return blocks,requests

def test_future_append_preserves_earlier_prefix():
    b,r=fixture();before=attach_prefixes(b,r)
    b.append({"user_raw":1,"timestamp_ms":400,**block_summary([99],[])})
    r.append({"user_raw":1,"timestamp_ms":400,"task":"recommendation"})
    assert attach_prefixes(b,r)[:len(before)]==before

def test_tied_permutation_and_duplicates():
    assert block_summary([1,2,1],[[5],[6]])==block_summary([2,1],[[6],[5]])

def test_score_whole_current_block_before_update():
    b,r=fixture();a=attach_prefixes(b,r)
    assert a[0]["history_ids"]==a[1]["history_ids"]==[0]
    assert a[0]["cohort"]=="transition"
    assert a[2]["cohort"]=="ambiguous" # immediately previous mixed block

def test_current_labels_do_not_change_inputs():
    b,r=fixture();a=attach_prefixes(b,r);changed=copy.deepcopy(r)
    changed[0]["labels"]=[0,1];c=attach_prefixes(b,changed)
    assert {k:v for k,v in a[0].items() if k!="labels"}=={k:v for k,v in c[0].items() if k!="labels"}

def test_exact_boundaries():
    assert [split_at(t) for t in CUTOFFS]==["validation","test","unused"]
    assert split_at(CUTOFFS[0]-1)=="train"

def test_gap_contract():
    assert gap_bucket(None)==0
    assert gap_bucket(60000)==1
    assert gap_bucket(60001)==2
    with pytest.raises(ValueError):gap_bucket(0)

def test_constrained_tokens():
    assert tokens('[]')==[]
    assert tokens('[1, 2]')==[1,2]
    for invalid in ("[1.0]","[-1]","[True]","__import__('os')", "[9223372036854775808]"):
        with pytest.raises(ValueError):tokens(invalid)


"""End-to-end pinned adapter fixtures: exact numbers, CRC and retained rows.

Tiny fake archives use all pinned release headers; monkeypatching MD5 only
permits fixtures through the release checksum gate. Production files are never
read or written by these tests.
"""
import csv
import copy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pyarrow.parquet as pq
import pyarrow as pa
import pytest

from unisar.data import audit


def fixture_archive(tmp_path, monkeypatch, *, rec_overrides=None, search_overrides=None,
                    metadata_overrides=None, duplicate_candidate=False, conflicting_group=False, unknown_header=False):
    rows = {
        "rec_inter.csv": [{"user_id": "1", "item_id": "100", "timestamp": "100.0", "click": "1"}],
        "src_inter.csv": [{"user_id": "1", "item_id": "100", "search_session_id": "1",
            "search_session_timestamp": "101.0", "keyword": "[2,3]", "click_cnt": "1.0",
            "item_type": "VIDEO", "search_source": "0", "search_session_time": "0"},
            {"user_id": "1", "item_id": "101", "search_session_id": "1",
            "search_session_timestamp": "101.0", "keyword": "[2,3]", "click_cnt": "0",
            "item_type": "VIDEO", "search_source": "0", "search_session_time": "0"}],
        "item_features.csv": [{"item_id": "100", "caption": "[2]", "author_id": "1",
            "first_level_category_id": "1", "music_id": "1", "first_level_category_name_en": "fixture"}],
        "user_features.csv": [{"user_id": "1", "reco_active_level": "1", "search_active_level": "1",
            "onehot_feat1": "1", "onehot_feat2": "1"}],
        "social_network.csv": [{"user_id": "1", "user_follow_id": "2"}],
    }
    if rec_overrides:
        rows["rec_inter.csv"][0].update(rec_overrides)
    if search_overrides:
        for row in rows["src_inter.csv"]:
            row.update(search_overrides)
    if metadata_overrides:
        rows["item_features.csv"][0].update(metadata_overrides)
    if duplicate_candidate:
        rows["src_inter.csv"][0]["click_cnt"] = "0"
        rows["src_inter.csv"].append({**rows["src_inter.csv"][0], "click_cnt": "2.0"})
    if conflicting_group:
        rows["src_inter.csv"].extend([
            {**rows["src_inter.csv"][0], "search_session_id": "99", "keyword": "[2]"},
            {**rows["src_inter.csv"][0], "search_session_id": "99", "keyword": "[3]"},
        ])
    if unknown_header:
        rows["rec_inter.csv"][0]["unexpected_future_field"] = "1"
    archive = tmp_path / "raw" / "fixture.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for name, values in rows.items():
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
            handle.writestr("KuaiSAR_final/" + name, stream.getvalue())
    monkeypatch.setattr(audit, "MD5", hashlib.md5(archive.read_bytes()).hexdigest())
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    return archive


@pytest.mark.parametrize("field,value", [
    ("click", "0.4"), ("click", "0.6"), ("click", "1.4"),
    ("timestamp", "100.0001"), ("timestamp", "1e2"),
    ("user_id", "1.4"), ("item_id", "100.4"),
])
def test_rec_numeric_fields_reject_rounding(tmp_path, monkeypatch, field, value):
    archive = fixture_archive(tmp_path, monkeypatch, rec_overrides={field: value})
    with pytest.raises(ValueError, match="Invalid required input"):
        audit.prepare(archive, tmp_path / "canonical", sizes=(1,))


@pytest.mark.parametrize("field,value", [
    ("click_cnt", "1.0001"), ("click_cnt", "1e2"),
    ("search_session_timestamp", "101.0001"),
    ("search_session_id", "1.4"), ("item_id", "100.4"),
])
def test_search_numeric_fields_reject_rounding(tmp_path, monkeypatch, field, value):
    archive = fixture_archive(tmp_path, monkeypatch, search_overrides={field: value})
    with pytest.raises(ValueError, match="Invalid required input"):
        audit.prepare(archive, tmp_path / "canonical", sizes=(1,))


def test_equal_length_corrupt_extraction_is_refreshed(tmp_path, monkeypatch):
    archive = fixture_archive(tmp_path, monkeypatch)
    destination = tmp_path / "extracted"
    audit.extract_and_inspect(archive, destination)
    out = destination / "rec_inter.csv"
    valid = out.read_bytes()
    corrupt = bytearray(valid)
    corrupt[-3] = ord("9") if corrupt[-3] != ord("9") else ord("8")
    assert len(valid) == len(corrupt)
    out.write_bytes(corrupt)
    audit.extract_and_inspect(archive, destination)
    assert out.read_bytes() == valid


def test_unknown_header_fails_before_projection(tmp_path, monkeypatch):
    archive = fixture_archive(tmp_path, monkeypatch, unknown_header=True)
    with pytest.raises(ValueError, match="Unexpected small-release header"):
        audit.extract_and_inspect(archive, tmp_path / "extracted")


def test_metadata_key_must_not_be_rounded(tmp_path, monkeypatch):
    archive = fixture_archive(tmp_path, monkeypatch, metadata_overrides={"item_id": "100.4"})
    with pytest.raises(ValueError, match="Invalid metadata key"):
        audit.prepare(archive, tmp_path / "canonical", sizes=(1,))


def test_missing_metadata_candidates_preserved_and_duplicates_max_binary(tmp_path, monkeypatch):
    archive = fixture_archive(tmp_path, monkeypatch, duplicate_candidate=True, conflicting_group=True)
    result = audit.prepare(archive, tmp_path / "canonical", sizes=(1,))
    canonical = tmp_path / "canonical" / "1"
    rows = pq.read_table(canonical / "search.parquet").to_pylist()
    assert len(rows) == 1
    assert rows[0]["item_raw"] == [100, 101]
    assert rows[0]["labels"] == [1, 0]
    assert rows[0]["raw_count"] == 3
    assert rows[0]["retained_raw_count"] == 3
    assert result["quarantined_groups"] == 1
    assert result["quarantined_rows"] == 2
    manifest = json.loads((canonical / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["missing_search_metadata_candidates"] == 1
    assert manifest["counts"]["retained_search_candidates"] == 2
    assert json.loads((canonical / "users.json").read_text(encoding="utf-8")) == [1]


def _canonical_feature_fixture(path, *, future=False, flip_labels=False):
    from unisar.common import digest, write_json
    path.mkdir()
    timestamp = audit.CUTOFFS[0]
    search = [{"user_raw": 1, "session_raw": 1, "timestamp_ms": timestamp - 10,
               "query_signature": "[3]", "item_raw": [7, 8],
               "labels": [0, 1] if flip_labels else [1, 0], "raw_count": 2, "retained_raw_count": 2}]
    rec = [{"user_raw": 1, "item_raw": 9, "timestamp_ms": timestamp - 20, "label": 1, "raw_row": 1}]
    metadata = [{"item_raw": item, "caption": f"[{token}]", "author_raw": 1,
                 "category_raw": 1, "category_name": "fixture"} for item, token in ((7, 11), (8, 12), (9, 13))]
    if future:
        # A lower raw user ID changes global block positions; encoded earlier
        # histories must still preserve identity content and model predictions.
        search.append({"user_raw": 0, "session_raw": 99, "timestamp_ms": timestamp + 10,
                       "query_signature": "[999]", "item_raw": [77], "labels": [1],
                       "raw_count": 1, "retained_raw_count": 1})
        metadata.append({"item_raw": 77, "caption": "[777]", "author_raw": 1,
                         "category_raw": 1, "category_name": "future-only"})
    for name, rows in (("search", search), ("recommendation", rec), ("metadata", metadata)):
        pq.write_table(pa.Table.from_pylist(rows), path / f"{name}.parquet")
    write_json(path / "users.json", [1])
    write_json(path / "manifest.json", {"data_hash": "fixture", "cohort_hash": "fixture",
        "cutoffs_utc8_ms": audit.CUTOFFS,
        "files": {name: digest(path / name) for name in ("search.parquet", "recommendation.parquet", "metadata.parquet", "users.json")}})




def test_future_append_and_outcomes_preserve_encoded_inputs(tmp_path, monkeypatch):
    import unisar.data.prepare as preparation
    from unisar.data.dataset import PreparedDataset
    monkeypatch.setattr(preparation, 'ROOT', tmp_path)
    datasets=[]
    for name,future,flip in [('base',False,False),('future',True,False),('labels',False,True)]:
        canonical=tmp_path/f'canonical-{name}'
        _canonical_feature_fixture(canonical,future=future,flip_labels=flip)
        preparation.build(canonical,tmp_path/name)
        datasets.append(PreparedDataset(tmp_path/name))
    assert datasets[0].vocab == datasets[1].vocab == datasets[2].vocab
    rows=[dataset.requests('train','search')[0] for dataset in datasets]
    def inputs(row):
        return {k:v for k,v in row.items() if k not in ('labels','history_ids')}
    assert inputs(rows[0]) == inputs(rows[1]) == inputs(rows[2])


def test_dataset_rejects_corrupt_prepared_file(tmp_path,monkeypatch):
    import unisar.data.prepare as preparation
    from unisar.data.dataset import PreparedDataset
    monkeypatch.setattr(preparation,'ROOT',tmp_path)
    canonical=tmp_path/'canonical'
    _canonical_feature_fixture(canonical)
    preparation.build(canonical,tmp_path/'prepared')
    with (tmp_path/'prepared'/'vocabulary.json').open('a',encoding='utf-8') as handle:
        handle.write(' ')
    with pytest.raises(ValueError,match='Prepared hash mismatch'):
        PreparedDataset(tmp_path/'prepared')


@pytest.mark.parametrize('sizes',[(),(0,),(-1,),(1,1)])
def test_invalid_cohort_sizes_fail_before_opening_archive(tmp_path,sizes):
    with pytest.raises(ValueError,match='distinct positive'):
        audit.prepare(tmp_path/'absent.zip',tmp_path/'canonical',sizes=sizes)
def test_preparation_preserves_existing_manifests(tmp_path):
    from unisar.data.prepare import build
    from unisar.data.audit import prepare
    import pytest
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "manifest.json").write_text('{"history":"preserve"}', encoding="utf-8")
    with pytest.raises(FileExistsError):
        build(tmp_path / "unused", output)
    canonical = tmp_path / "canonical/10000"
    canonical.mkdir(parents=True)
    (canonical / "manifest.json").write_text('{"history":"preserve"}', encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "unused.zip", tmp_path / "canonical", sizes=(10000,))
    assert (output / "manifest.json").read_text() == '{"history":"preserve"}'
    assert (canonical / "manifest.json").read_text() == '{"history":"preserve"}'
