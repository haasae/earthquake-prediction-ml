from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm


@dataclass(frozen=True)
class DatasetConfig:
    url: str
    raw_path: Path


def download_file(url: str, dest: Path, overwrite: bool = False, timeout: int = 60) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        return dest

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        chunk_size = 1024 * 64

        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return dest
