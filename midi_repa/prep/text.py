"""Caption features for training: T5 (77 x 1024) and CLAP (512) of each track's caption."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from midi_repa.data import read_jsonl, sharded_path
from midi_repa.text import TextEncoder


def build(root: Path, clap_checkpoint: Path | None = None, batch_size: int = 32) -> None:
    rows = {}
    for split in ("train", "validation"):
        for row in read_jsonl(root / "manifests" / f"{split}.jsonl"):
            rows[row["clip_id"]] = row
    pending = [rows[k] for k in sorted(rows, key=int)
               if not sharded_path(root / "text", k, ".npz").is_file()]
    if not pending:
        print("text features: all present")
        return
    encoder = TextEncoder("cuda", clap_checkpoint)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        t5, clap = encoder([row["caption"] for row in batch])
        t5, clap = t5.cpu().numpy(), clap.cpu().numpy()
        for index, row in enumerate(batch):
            destination = sharded_path(root / "text", row["clip_id"], ".npz")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez(handle, text_features=t5[index], text_features_c=clap[index])
            os.replace(temporary, destination)
        print(f"text features {min(start + batch_size, len(pending))}/{len(pending)}", flush=True)
