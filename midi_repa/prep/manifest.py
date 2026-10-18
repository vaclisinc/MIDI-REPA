"""Training/validation manifests from the JamendoMIDI splits, the latents and the captions.

Per track (paper Sec. 2.2.3):

* one of the two ATTM caption sources, fixed for the whole run:
  ``sha256(f"42:{id}")[0]`` even -> Qwen, odd -> Music Flamingo;
* train tracks contribute every complete 10-second window;
* each of the 444 monitoring tracks contributes one window,
  ``int(sha256(f"42:{id}:window")[:8]) % complete_windows``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from midi_repa.data import WINDOW_SAMPLES, sharded_path
from midi_repa.prep.jamendomidi import read_splits

SEED = 42
SOURCES = ("qwen", "music_flamingo")


def caption_source(clip_id: str, seed: int = SEED) -> str:
    return "qwen" if hashlib.sha256(f"{seed}:{clip_id}".encode()).digest()[0] % 2 == 0 \
        else "music_flamingo"


def validation_window(clip_id: str, complete_windows: int, seed: int = SEED) -> int:
    digest = hashlib.sha256(f"{seed}:{clip_id}:window".encode()).digest()[:8]
    return int.from_bytes(digest, "big") % complete_windows


def complete_windows(latent_path: Path) -> int:
    """Number of complete 10-second windows; they form a prefix of the cached windows."""
    with safe_open(latent_path, framework="np") as cached:
        valid = cached.get_tensor("valid_samples")
    full = np.flatnonzero(valid == WINDOW_SAMPLES).tolist()
    if full != list(range(len(full))) or not full:
        raise ValueError(f"complete windows are not a non-empty prefix: {latent_path}")
    return len(full)


def load_captions(path: Path) -> dict[str, str]:
    """ATTM caption file: a JSON list of ``{"path": ".../<id>.mp3", "caption": ...}``."""
    captions = {}
    for row in json.loads(Path(path).read_text(encoding="utf-8")):
        captions[Path(row["path"]).stem] = str(row["caption"])
    return captions


def build(root: Path, qwen_captions: Path, flamingo_captions: Path) -> None:
    captions = {"qwen": load_captions(qwen_captions),
                "music_flamingo": load_captions(flamingo_captions)}
    out = {"train": [], "validation": []}
    for split_row in read_splits(root):
        split, clip_id = split_row["split"], split_row["track_id"]
        if split not in out:
            continue  # SDD / ATTM reference tracks are never trained on
        source = caption_source(clip_id)
        windows = complete_windows(sharded_path(root / "latents", clip_id, ".safetensors"))
        out[split].append({
            "clip_id": clip_id,
            "split": split,
            "caption": captions[source][clip_id],
            "caption_source": source,
            "full_window_count": windows,
            "window_indices": [validation_window(clip_id, windows)] if split == "validation"
            else [],
        })
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    for split, rows in out.items():
        rows.sort(key=lambda row: int(row["clip_id"]))
        with (root / "manifests" / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"{split}: {len(rows)} tracks, "
              f"{sum(r['full_window_count'] if split == 'train' else 1 for r in rows)} windows")
