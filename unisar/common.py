"""Small, deterministic artifact utilities."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = ROOT / ".local"
DATA_ROOT = ROOT / "data"
ARTIFACTS_ROOT = ROOT / "artifacts"
WORK_ROOT = LOCAL_ROOT / "work"
REPORTS_ROOT = LOCAL_ROOT / "reports"
ENVIRONMENT_LOCK = LOCAL_ROOT / "environment" / "environment-lock.txt"


def reports_dir(root: Path = ROOT, date: str | dt.date | None = None) -> Path:
    """Private report directory for the Asia/Jakarta calendar date.

    Jakarta uses UTC+7; a fixed offset keeps this independent of system timezone
    settings and optional timezone database installations. An explicit date is
    useful when revisiting a historical study without changing its evidence.
    """
    if date is None:
        date = dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).date()
    elif isinstance(date, str):
        date = dt.date.fromisoformat(date)
    elif isinstance(date, dt.datetime):
        date = date.date()
    if not isinstance(date, dt.date):
        raise TypeError("Report date must be a date or an ISO YYYY-MM-DD string")
    return Path(root) / ".local" / "reports" / date.isoformat()

def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
