"""Acquire the pinned public release with corruption and identity checks."""
from __future__ import annotations
import argparse
import datetime as dt
import time
import urllib.request
from pathlib import Path
from unisar.common import DATA_ROOT, digest, write_json

URL = "https://zenodo.org/records/8181109/files/KuaiSAR.zip?download=1"
MD5 = "daea8cbf605db6bd5841740f0e4a12d9"
SIZE = 488077651

def acquire(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "KuaiSAR.zip"
    partial = directory / "KuaiSAR.zip.partial"
    if not target.exists():
        start = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "unisar-pytorch-academic-data-preparation/0.1"}
        if start:
            headers["Range"] = f"bytes={start}-"
        request = urllib.request.Request(URL, headers=headers)
        with urllib.request.urlopen(request, timeout=90) as response:
            status = response.getcode()
            if start and status != 206:
                start = 0
            elif start and not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                raise RuntimeError("Unexpected resumed Content-Range")
            size = start
            last = time.monotonic()
            with partial.open("ab" if start else "wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                    size += len(chunk)
                    if time.monotonic() - last >= 10:
                        print(f"download {size}/{SIZE} bytes ({100*size/SIZE:.1f}%)", flush=True)
                        last = time.monotonic()
        if partial.stat().st_size != SIZE or digest(partial, "md5") != MD5:
            raise RuntimeError("Release size/MD5 mismatch; preserve partial for inspection")
        partial.replace(target)
    if target.stat().st_size != SIZE or digest(target, "md5") != MD5:
        raise RuntimeError("Existing release fails identity check")
    manifest = {"url": URL, "acquired_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "bytes": SIZE, "md5": MD5, "sha256": digest(target), "license_url": "https://github.com/KuaiSAR/KuaiSAR.github.io/blob/main/LICENSE", "dataset_license": "CC-BY-NC-SA-4.0 (official project notice; redistribution reviewed separately)"}
    write_json(directory / "raw-release.json", manifest)
    print(f"verified {target}; sha256={manifest['sha256']}", flush=True)
    return target

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "--directory", dest="directory", type=Path, default=DATA_ROOT / "raw")
    args = parser.parse_args()
    acquire(args.directory)
