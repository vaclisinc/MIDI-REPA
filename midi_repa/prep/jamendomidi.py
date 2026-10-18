"""Download JamendoMIDI (huggingface.co/datasets/vaclis/JamendoMIDI) as MIDI files + split list.

    python -m midi_repa.prep jamendomidi --root data/

Writes ``<root>/midi/<shard>/<id>.mid`` and ``<root>/splits.csv`` with columns
``track_id, split, families, duration_s``. The splits are the paper's: 49,034 training
tracks, 444 loss-monitoring tracks, and 1,599 held-out test tracks that are never trained on.
"""

from __future__ import annotations

import csv
from pathlib import Path

from midi_repa.data import sharded_path

REPO_ID = "vaclis/JamendoMIDI"
EXPECTED = {"train": 49_034, "validation": 444, "test": 1_599}


def download(root: Path, revision: str | None = None) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(f for f in HfApi().list_repo_files(REPO_ID, repo_type="dataset",
                                                      revision=revision)
                   if f.startswith("data/") and f.endswith(".parquet"))
    root.mkdir(parents=True, exist_ok=True)
    rows, counts = [], {}
    for name in files:
        local = hf_hub_download(REPO_ID, name, repo_type="dataset", revision=revision)
        for record in pq.read_table(local).to_pylist():
            track_id = str(record["track_id"])
            destination = sharded_path(root / "midi", track_id, ".mid")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(record["midi"])
            rows.append({"track_id": track_id, "split": record["split"],
                         "families": "|".join(record["families"]),
                         "duration_s": record["duration_s"]})
            counts[record["split"]] = counts.get(record["split"], 0) + 1
        print(f"{name}: {len(rows)} tracks so far", flush=True)
    if counts != EXPECTED:
        raise SystemExit(f"unexpected split sizes {counts}, expected {EXPECTED}")
    rows.sort(key=lambda row: int(row["track_id"]))
    with (root / "splits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} MIDI files and {root / 'splits.csv'}: {counts}")


def read_splits(root: Path) -> list[dict]:
    with (Path(root) / "splits.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
