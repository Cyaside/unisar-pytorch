"""Pinned KuaiSAR-small adapter, complete-log audit and deterministic cohorts.

DuckDB scans typed projections from the raw CSVs; only selected item metadata
is retained. All raw files remain immutable and outside Git.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import io
import json
import re
import time
import zipfile
import zlib
from pathlib import Path
import duckdb
from unisar.common import DATA_ROOT, ROOT, digest, object_hash, read_json, reports_dir, write_json
from unisar.data.download import MD5

TZ = dt.timezone(dt.timedelta(hours=8))
CUTOFFS = [int(dt.datetime(2023, 5, day, tzinfo=TZ).timestamp()*1000) for day in (29,30,31)]
REQUIRED = {
    "rec_inter.csv": {"user_id","item_id","timestamp","click"},
    "src_inter.csv": {"user_id","search_session_id","search_session_timestamp","keyword","item_id","click_cnt","item_type","search_source","search_session_time"},
    "item_features.csv": {"item_id","caption","author_id","first_level_category_id","music_id"},
    "user_features.csv": {"user_id","reco_active_level","search_active_level","onehot_feat1","onehot_feat2"},
    "social_network.csv": {"user_id","user_follow_id"},
}
ALLOWED = {
    "rec_inter.csv": REQUIRED["rec_inter.csv"] | {"playing_time","duration_ms","time","forward","like","follow","search","search_item_related"},
    "src_inter.csv": REQUIRED["src_inter.csv"],
    "item_features.csv": REQUIRED["item_features.csv"] | {"item_type","upload_time","upload_type","second_level_category_id","third_level_category_id","fourth_level_category_id"} | {f"{level}_level_category_name{suffix}" for level in ("first","second","third","fourth") for suffix in ("","_en")},
    "user_features.csv": REQUIRED["user_features.csv"],
    "social_network.csv": REQUIRED["social_network.csv"],
}

def integral_text(value: str | None, binary: bool = False) -> int:
    """Pinned release uses decimal integer spelling, optionally zero fraction.

    Reject fractional precision before conversion; SQL casts may otherwise round.
    Scientific notation is intentionally unsupported in this release adapter.
    """
    if value is None or not re.fullmatch(r"[0-9]+(?:\.0+)?",value):
        raise ValueError("Expected nonnegative decimal integral field")
    result=int(value.split('.')[0])
    if result>2**63-1 or (binary and result not in (0,1)):
        raise ValueError("Integer outside declared domain")
    return result

def file_crc32(path:Path) -> int:
    crc=0
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):
            crc=zlib.crc32(chunk,crc)
    return crc & 0xffffffff

def tokens(value: str | None) -> list[int]:
    if value is None:
        return []
    if not re.fullmatch(r"\[\s*(?:\d+\s*(?:,\s*\d+\s*)*)?\]", value):
        raise ValueError(f"Invalid integer token list: {value[:100]!r}")
    result = json.loads(value)
    if any(type(x) is not int or x < 0 or x > 2**63-1 for x in result):
        raise ValueError("Invalid token integer")
    return result

def sqlpath(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"

def extract_and_inspect(archive: Path, destination: Path) -> dict:
    if digest(archive, "md5") != MD5:
        raise ValueError("Unrecognized release MD5")
    destination.mkdir(parents=True, exist_ok=True)
    entries, headers = [], {}
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            entries.append({"name": info.filename, "bytes": info.file_size, "compressed_bytes": info.compress_size, "crc32": info.CRC})
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if name not in REQUIRED or name in headers:
                raise ValueError(f"Unexpected/duplicate archive entry {info.filename}")
            with z.open(info) as stream:
                header = next(csv.reader(io.TextIOWrapper(stream, encoding="utf-8-sig")))
            if len(header) != len(set(header)) or not REQUIRED[name] <= set(header) or not set(header) <= ALLOWED[name]:
                raise ValueError(f"Unexpected small-release header for {name}: {header}")
            headers[name] = header
            out = destination / name
            if not out.exists() or out.stat().st_size != info.file_size or file_crc32(out)!=info.CRC:
                print(f"extracting {name}: {info.file_size:,} bytes", flush=True)
                with z.open(info) as stream, out.open("wb") as f:
                    for chunk in iter(lambda: stream.read(4*1024*1024), b""):
                        f.write(chunk)
    if set(headers) != set(REQUIRED):
        raise ValueError("Missing release file")
    return {"archive_entries": entries, "headers": headers, "schema_hash": object_hash(headers)}

def prepare(archive: Path, output: Path, sizes=(200,2000)) -> dict:
    sizes = tuple(int(size) for size in sizes)
    if not sizes or any(size < 1 for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("Cohort sizes must be distinct positive integers")
    for size in sizes:
        name = "smoke" if size == 200 else "pilot" if size == 2000 else str(size)
        if (output / name / "manifest.json").exists():
            raise FileExistsError("Canonical manifest exists; choose a new output root to preserve its history.")
    started = time.perf_counter()
    raw = archive.parent / "extracted"
    audit = extract_and_inspect(archive, raw)
    output.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect(str(output / "audit.duckdb"))
    db.execute("SET threads=4")
    db.execute("SET memory_limit='4GB'")
    db.execute("SET preserve_insertion_order=true")
    def scalar(sql): return db.execute(sql).fetchone()[0]
    def source(name): return f"read_csv({sqlpath(raw/name)}, header=true, all_varchar=true, sample_size=20480)"
    for name in REQUIRED:
        view = name.replace(".csv", "")
        db.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM {source(name)}")
    print("auditing identifiers, timestamps, labels and token signatures", flush=True)
    audit["raw_rows"] = {name: scalar(f"SELECT count(*) FROM {name.replace('.csv','')}") for name in REQUIRED}
    invalid_rec = scalar("""SELECT count(*) FROM rec_inter WHERE
        try_cast(user_id AS BIGINT) IS NULL OR NOT regexp_full_match(user_id,'[0-9]+') OR
        try_cast(item_id AS BIGINT) IS NULL OR NOT regexp_full_match(item_id,'[0-9]+') OR
        try_cast(timestamp AS BIGINT) IS NULL OR NOT regexp_full_match(timestamp,'[0-9]+(?:[.]0+)?') OR
        try_cast(click AS INTEGER) IS NULL OR NOT regexp_full_match(click,'(?:0|1)(?:[.]0+)?')""")
    invalid_search = scalar("""SELECT count(*) FROM src_inter WHERE
        try_cast(user_id AS BIGINT) IS NULL OR NOT regexp_full_match(user_id,'[0-9]+') OR
        try_cast(item_id AS BIGINT) IS NULL OR NOT regexp_full_match(item_id,'[0-9]+') OR
        try_cast(search_session_id AS BIGINT) IS NULL OR NOT regexp_full_match(search_session_id,'[0-9]+') OR
        try_cast(search_session_timestamp AS BIGINT) IS NULL OR NOT regexp_full_match(search_session_timestamp,'[0-9]+(?:[.]0+)?') OR
        try_cast(click_cnt AS BIGINT) IS NULL OR NOT regexp_full_match(click_cnt,'[0-9]+(?:[.]0+)?') OR keyword IS NULL OR
        NOT regexp_full_match(keyword, '\\[\\s*(?:\\d+\\s*(?:,\\s*\\d+\\s*)*)?\\]') OR item_type IS NULL""")
    audit["invalid_required_rows"] = {"recommendation": invalid_rec, "search": invalid_search}
    if invalid_rec or invalid_search:
        write_json(output / "audit-failed.json", audit)
        raise ValueError("Invalid required input rows; inspect audit-failed.json before continuing")
    print("projecting raw interactions to bounded typed tables", flush=True)
    db.execute("""CREATE OR REPLACE TABLE rec AS SELECT cast(user_id AS BIGINT) user_raw, cast(item_id AS BIGINT) item_raw,
        cast(timestamp AS BIGINT) timestamp_ms, cast(click AS TINYINT) AS label,
        row_number() OVER (ORDER BY cast(user_id AS BIGINT),cast(timestamp AS BIGINT),cast(item_id AS BIGINT),cast(click AS TINYINT)) raw_row FROM rec_inter""")
    db.execute("""CREATE OR REPLACE TABLE search_raw AS SELECT cast(user_id AS BIGINT) user_raw, cast(search_session_id AS BIGINT) session_raw,
        cast(search_session_timestamp AS BIGINT) timestamp_ms, regexp_replace(keyword,'\\s','','g') query_signature,
        cast(item_id AS BIGINT) item_raw, (cast(click_cnt AS BIGINT)>0) AS label, item_type FROM src_inter""")
    db.execute("""CREATE OR REPLACE TABLE search_groups AS SELECT user_raw,session_raw,min(timestamp_ms) timestamp_ms,
        min(query_signature) query_signature,count(*) raw_count,count(DISTINCT item_raw) distinct_items,
        (count(DISTINCT timestamp_ms)>1 OR count(DISTINCT query_signature)>1) AS conflict
        FROM search_raw GROUP BY user_raw,session_raw""")
    audit["search_groups"] = scalar("SELECT count(*) FROM search_groups")
    audit["quarantined_groups"] = scalar("SELECT count(*) FROM search_groups WHERE conflict")
    audit["quarantined_rows"] = scalar("SELECT coalesce(sum(raw_count),0) FROM search_groups WHERE conflict")
    audit["search_type_counts"] = dict(db.execute("SELECT item_type,count(*) FROM search_raw GROUP BY item_type ORDER BY item_type").fetchall())
    audit["search_type_positives"] = dict(db.execute("SELECT item_type,sum(cast(label AS INTEGER)) FROM search_raw GROUP BY item_type ORDER BY item_type").fetchall())
    audit["timestamp_range_ms"] = {table: db.execute(f"SELECT min(timestamp_ms),max(timestamp_ms) FROM {table}").fetchone() for table in ("rec","search_raw")}
    audit["recommendation_labels"] = dict(db.execute("SELECT label,count(*) FROM rec GROUP BY label").fetchall())
    audit["recommendation_duplicate_exposures"] = scalar("SELECT count(*) FROM rec") - scalar("SELECT count(*) FROM (SELECT DISTINCT user_raw,timestamp_ms,item_raw FROM rec)")
    db.execute("""CREATE OR REPLACE TABLE search_kept AS SELECT r.user_raw,r.session_raw,r.item_raw,max(cast(r.label AS TINYINT)) AS label,
        count(*) multiplicity FROM search_raw r JOIN search_groups g USING(user_raw,session_raw)
        WHERE NOT g.conflict AND r.item_type IN ('VIDEO','IMAGE_ATLAS') GROUP BY r.user_raw,r.session_raw,r.item_raw""")
    train_end = CUTOFFS[0]
    db.execute(f"""CREATE OR REPLACE TABLE eligible AS SELECT DISTINCT g.user_raw FROM search_groups g JOIN search_kept k USING(user_raw,session_raw)
        WHERE g.timestamp_ms < {train_end} AND NOT g.conflict AND g.user_raw IN (SELECT user_raw FROM rec WHERE timestamp_ms < {train_end})""")
    users = [r[0] for r in db.execute("SELECT user_raw FROM eligible").fetchall()]
    import hashlib
    users.sort(key=lambda u: hashlib.sha256(f"42:{u}".encode()).hexdigest())
    audit["eligible_users"] = len(users)
    audit["raw_users"] = scalar("SELECT count(*) FROM (SELECT user_raw FROM rec UNION SELECT user_raw FROM search_raw)")
    audit["training_ineligible_users"] = audit["raw_users"] - len(users)
    if len(users) < max(sizes):
        raise ValueError("Insufficient eligible users for requested fixed cohorts")
    selected = users[:max(sizes)]
    db.execute("CREATE OR REPLACE TABLE selected(user_raw BIGINT)")
    db.executemany("INSERT INTO selected VALUES (?)", [(u,) for u in selected])
    db.execute("""CREATE OR REPLACE TABLE selected_search AS SELECT g.user_raw,g.session_raw,g.timestamp_ms,g.query_signature,
        coalesce(list(k.item_raw ORDER BY k.item_raw) FILTER(WHERE k.item_raw IS NOT NULL),[]::BIGINT[]) item_raw,
        coalesce(list(k.label ORDER BY k.item_raw) FILTER(WHERE k.item_raw IS NOT NULL),[]::TINYINT[]) labels,
        g.raw_count,coalesce(sum(k.multiplicity),0) retained_raw_count
        FROM search_groups g JOIN selected u USING(user_raw) LEFT JOIN search_kept k USING(user_raw,session_raw)
        WHERE NOT g.conflict GROUP BY g.user_raw,g.session_raw,g.timestamp_ms,g.query_signature,g.raw_count""")
    db.execute("""CREATE OR REPLACE TABLE selected_rec AS SELECT r.* FROM rec r JOIN selected USING(user_raw)""")
    db.execute("""CREATE OR REPLACE TABLE needed_items AS SELECT DISTINCT item_raw FROM selected_rec
        UNION SELECT DISTINCT unnest(item_raw) FROM selected_search""")
    print("semi-joining released metadata to selected candidates/history IDs", flush=True)
    invalid_metadata = scalar("SELECT count(*) FROM item_features WHERE try_cast(item_id AS BIGINT) IS NULL OR NOT regexp_full_match(item_id,'[0-9]+')")
    if invalid_metadata:
        raise ValueError("Invalid metadata key")
    db.execute("""CREATE OR REPLACE TABLE metadata AS SELECT cast(m.item_id AS BIGINT) item_raw,m.caption,
        try_cast(m.author_id AS BIGINT) author_raw,try_cast(m.first_level_category_id AS BIGINT) category_raw,
        m.first_level_category_name_en category_name FROM item_features m JOIN needed_items n ON cast(m.item_id AS BIGINT)=n.item_raw""")
    if scalar("SELECT count(*) FROM metadata") != scalar("SELECT count(DISTINCT item_raw) FROM metadata"):
        raise ValueError("Duplicate metadata keys")
    captions = db.execute("SELECT item_raw,caption FROM metadata").fetchall()
    for _, caption in captions:
        tokens(caption)
    audit["selected_metadata_items"] = len(captions)
    audit["needed_items"] = scalar("SELECT count(*) FROM needed_items")
    audit["missing_metadata_items"] = audit["needed_items"] - len(captions)
    cohort_manifests = {}
    for size in sizes:
        cohort_name = "smoke" if size == 200 else "pilot" if size == 2000 else str(size)
        dest = output / cohort_name
        dest.mkdir(parents=True, exist_ok=True)
        ordered = users[:size]
        write_json(dest / "users.json", ordered)
        db.execute("CREATE OR REPLACE TEMP TABLE cohort_users(user_raw BIGINT)")
        db.executemany("INSERT INTO cohort_users VALUES (?)", [(u,) for u in ordered])
        for table, filename in (("selected_search","search.parquet"),("selected_rec","recommendation.parquet")):
            order = "user_raw,timestamp_ms,session_raw" if table == "selected_search" else "user_raw,timestamp_ms,item_raw,raw_row"
            db.execute(f"COPY (SELECT t.* FROM {table} t JOIN cohort_users USING(user_raw) ORDER BY {order}) TO {sqlpath(dest/filename)} (FORMAT PARQUET, COMPRESSION ZSTD)")
        db.execute(f"COPY (SELECT * FROM metadata ORDER BY item_raw) TO {sqlpath(dest/'metadata.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)")
        counts = {"search_groups": scalar("SELECT count(*) FROM selected_search JOIN cohort_users USING(user_raw)"), "recommendation_exposures": scalar("SELECT count(*) FROM selected_rec JOIN cohort_users USING(user_raw)")}
        counts["retained_search_candidates"] = scalar("SELECT coalesce(sum(len(item_raw)),0) FROM selected_search JOIN cohort_users USING(user_raw)")
        counts["original_search_rows"] = scalar("SELECT coalesce(sum(raw_count),0) FROM selected_search JOIN cohort_users USING(user_raw)")
        counts["retained_search_rows_before_dedup"] = scalar("SELECT coalesce(sum(retained_raw_count),0) FROM selected_search JOIN cohort_users USING(user_raw)")
        counts["search_empty_after_type_filter"] = scalar("SELECT count(*) FROM selected_search JOIN cohort_users USING(user_raw) WHERE len(item_raw)=0")
        counts["original_search_positive_rows"] = scalar("SELECT coalesce(sum(cast(r.label AS BIGINT)),0) FROM search_raw r JOIN cohort_users USING(user_raw) JOIN search_groups g USING(user_raw,session_raw) WHERE NOT g.conflict")
        counts["retained_search_positive_rows"] = scalar("SELECT coalesce(sum(cast(r.label AS BIGINT)),0) FROM search_raw r JOIN cohort_users USING(user_raw) JOIN search_groups g USING(user_raw,session_raw) WHERE NOT g.conflict AND r.item_type IN ('VIDEO','IMAGE_ATLAS')")
        counts["missing_search_metadata_candidates"] = scalar("SELECT count(*) FROM (SELECT unnest(item_raw) item_raw FROM selected_search JOIN cohort_users USING(user_raw)) c LEFT JOIN metadata m USING(item_raw) WHERE m.item_raw IS NULL")
        counts["missing_recommendation_metadata_exposures"] = scalar("SELECT count(*) FROM selected_rec r JOIN cohort_users USING(user_raw) LEFT JOIN metadata m USING(item_raw) WHERE m.item_raw IS NULL")
        files = {n: digest(dest/n) for n in ("users.json","search.parquet","recommendation.parquet","metadata.parquet")}
        manifest = {"schema_version":"kuaisar-small-v1", "raw_archive_sha256":digest(archive),"schema_hash":audit["schema_hash"],"cohort_size":size,"cohort_hash":object_hash(ordered),"selection_rule":"sha256 UTF8 42:<raw_user_id>, both training services", "cutoffs_utc8_ms":CUTOFFS,"counts":counts,"files":files}
        manifest["data_hash"] = object_hash(manifest)
        write_json(dest / "manifest.json", manifest)
        cohort_manifests[cohort_name] = manifest
    audit["cohorts"] = cohort_manifests
    audit["cutoffs_utc8_ms"] = CUTOFFS
    audit["elapsed_seconds"] = time.perf_counter()-started
    write_json(output / "data-audit.json", audit)
    report_name = "data-audit.md" if output.resolve() == (DATA_ROOT / "canonical").resolve() else f"data-audit-{output.name}.md"
    report = reports_dir(ROOT) / report_name
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# KuaiSAR-small local data audit\n\nComplete release integrity verified; canonical cohort files are outside Git.\n\n```json\n"+json.dumps(audit,indent=2)+"\n```\n\nMetadata is a released static snapshot. Search VIDEO/IMAGE_ATLAS candidates use left joins; missing metadata never removes a candidate. Excluded candidate types affect the evaluation population. Recommendation duplicate exposures are retained as recorded. Empty filtered search sessions remain issued-query history. Inconsistent search sessions are quarantined, with counts above. User/social snapshots are audited by header/row count and excluded from features.\n",encoding="utf-8")
    db.close()
    print(f"prepared canonical cohorts in {time.perf_counter()-started:.1f}s", flush=True)
    return audit

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive",type=Path,default=DATA_ROOT/"raw/KuaiSAR.zip")
    parser.add_argument("--output",type=Path,default=DATA_ROOT/"canonical")
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 2000],
                        help="Nested cohort sizes selected exclusively from training-eligible users")
    args=parser.parse_args()
    prepare(args.archive,args.output,tuple(args.sizes))
