"""Fetch the Kaggle dataset without credentials (the download endpoint is public),
falling back to a Hugging Face mirror. Writes data/raw/twcs.csv and nothing else."""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from support_agent import config

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/thoughtvector/customer-support-on-twitter"
HF_URL = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"
RAW_CSV = config.RAW_DIR / "twcs.csv"


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("Content-Length") or 0)
        buf, read = io.BytesIO(), 0
        while chunk := r.read(1 << 20):
            buf.write(chunk)
            read += len(chunk)
            if total:
                print(f"\r  {read / 1e6:6.1f} / {total / 1e6:6.1f} MB", end="", file=sys.stderr)
        print(file=sys.stderr)
        return buf.getvalue()


def ensure_raw() -> Path:
    if RAW_CSV.exists():
        return RAW_CSV
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        print("Downloading Kaggle zip ...", file=sys.stderr)
        data = _fetch(KAGGLE_URL)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            name = next(n for n in zf.namelist() if n.endswith("twcs.csv"))
            RAW_CSV.write_bytes(zf.read(name))
    except Exception as exc:  # noqa: BLE001 - any failure -> try mirror
        print(f"Kaggle download failed ({exc}); trying Hugging Face mirror ...", file=sys.stderr)
        RAW_CSV.write_bytes(_fetch(HF_URL))
    return RAW_CSV


if __name__ == "__main__":
    print(ensure_raw())
