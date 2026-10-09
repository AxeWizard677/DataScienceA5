"""Download the Leyenda Deliverable-1 dataset from the CESI Cloudreve share into data/raw/.

Usage:  uv run python scripts/download_dataset.py [--workers 8] [--categories Photo Text]

Resumable: files already present with the right size are skipped, so re-running only
fetches what is missing or was corrupted.
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SERVER = "https://file.legionera.fr"
SHARE = "cloudreve://jAFr:datascience@share"  # complete dataset (10,000 Schematics), password-protected share
CATEGORIES = ["Painting", "Photo", "Schematics", "Sketch", "Text"]
DEST = Path(__file__).resolve().parents[1] / "data" / "raw"
HEADERS = {"User-Agent": "leyenda-downloader/1.0", "Content-Type": "application/json"}


def api(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(SERVER + path, data=data, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.load(r)
    if body.get("code") != 0:
        raise RuntimeError(f"{path}: {body.get('msg')}")
    return body["data"]


def list_files(uri):
    token = None
    while True:
        q = f"/api/v4/file?page_size=2000&uri={urllib.parse.quote(uri, safe='')}"
        if token:
            q += f"&next_page_token={urllib.parse.quote(token)}"
        d = api(q)
        for f in d["files"]:
            if f["type"] == 1:  # folder
                yield from list_files(f["path"])
            else:
                yield f
        token = d["pagination"].get("next_token")
        if not token:
            return


def download(url, target, size, retries=3):
    tmp = target.with_suffix(target.suffix + ".part")
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": HEADERS["User-Agent"]})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as out:
                while chunk := r.read(1 << 16):
                    out.write(chunk)
            if tmp.stat().st_size == size:
                tmp.rename(target)
                return True
        except Exception:
            pass
    tmp.unlink(missing_ok=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=256)
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    args = ap.parse_args()

    todo = []
    for cat in args.categories:
        files = list(list_files(f"{SHARE}/{cat}"))
        (DEST / cat).mkdir(parents=True, exist_ok=True)
        missing = [f for f in files
                   if not ((DEST / cat / f["name"]).exists()
                           and (DEST / cat / f["name"]).stat().st_size == f["size"])]
        print(f"{cat:11s} {len(files):6d} files, {len(missing):6d} to download", flush=True)
        todo += [(cat, f) for f in missing]

    # The server answers each request after ~5 s whatever the file size, so throughput comes
    # from many concurrent requests. Signed URLs are fetched in batches by a separate pool so
    # downloads never wait on them (no per-batch barrier).
    def sign(batch):
        urls = api("/api/v4/file/url", {"uris": [f["path"] for _, f in batch], "download": True})["urls"]
        return batch, urls

    failed, done = [], 0
    batches = [todo[i:i + 100] for i in range(0, len(todo), 100)]
    with ThreadPoolExecutor(4) as signer, ThreadPoolExecutor(args.workers) as pool:
        futures = {}
        for signed in as_completed([signer.submit(sign, b) for b in batches]):
            batch, urls = signed.result()
            for (cat, f), u in zip(batch, urls):
                futures[pool.submit(download, u["url"], DEST / cat / f["name"], f["size"])] = f["path"]
        for fut in as_completed(futures):
            done += 1
            if not fut.result():
                failed.append(futures[fut])
            if done % 500 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} downloaded, {len(failed)} failed", flush=True)

    if failed:
        print("FAILED (re-run the script to retry):", *failed[:20], sep="\n  ")
        sys.exit(1)
    print("All files downloaded to", DEST)


if __name__ == "__main__":
    main()
