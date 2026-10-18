"""Training data: one complete 10-second window per item.

On-disk layout under ``data_root`` (written by :mod:`midi_repa.prep`)::

    manifests/{train,validation}.jsonl   one row per track
    latents/<shard>/<id>.safetensors     SAME-L latents  fp16 [windows, 256, 108]
    text/<shard>/<id>.npz                T5 [77, 1024] + CLAP [512] caption features
    teacher/<name>/<shard>/<id>.npz      packed binary rolls [windows, 108, ceil(D/8)]

``<shard>`` is ``sha1(f"{id}.mp3")[:2]``. Training rows contribute every complete window
of the track; each validation row contributes the single window in ``window_indices``.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

WINDOW_SAMPLES = 441_000  # 10 s at 44.1 kHz
LATENT_FRAMES = 108
LATENT_DIM = 256
PITCH_MIN, PITCH_MAX = 21, 108
PITCHES = PITCH_MAX - PITCH_MIN + 1  # 88
LATENT_LAYOUT = ("latent_layout", "c_contiguous_v2")

# Symbolic targets of the paper: name -> number of instrument slices on the 88-pitch axis.
TEACHERS = {
    "muscriptor_roll": 1,   # MuScriptor transcription, instruments merged (108 x 88)
    "muscriptor_cube": 7,   # MuScriptor 7-family cube, flattened class-major (108 x 616)
    "basic_pitch_roll": 1,  # Basic Pitch transcription (108 x 88)
}


def shard(clip_id: str) -> str:
    return hashlib.sha1(f"{clip_id}.mp3".encode()).hexdigest()[:2]


def sharded_path(root: Path, clip_id: str, suffix: str) -> Path:
    return Path(root) / shard(clip_id) / f"{clip_id}{suffix}"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


class JamendoWindows(Dataset):
    """Items: ``latent`` (108, 256), ``text_features`` (77, 1024), ``text_features_c`` (512),
    and, when a teacher is configured, ``zs`` (108, slices * 88) as float32."""

    def __init__(self, manifest: Path, split: str, *, latent_dir: Path, text_dir: Path,
                 teacher_dir: Path | None = None, teacher_slices: int = 1):
        self.rows = read_jsonl(manifest)
        if not self.rows:
            raise ValueError(f"empty manifest: {manifest}")
        self.split = split
        self.latent_dir = Path(latent_dir)
        self.text_dir = Path(text_dir)
        self.teacher_dir = Path(teacher_dir) if teacher_dir is not None else None
        self.teacher_dim = teacher_slices * PITCHES

        self.cumulative: list[int] = []
        total = 0
        for row in self.rows:
            if row["split"] != split:
                raise ValueError(f"row {row['clip_id']} is {row['split']}, expected {split}")
            count = (int(row["full_window_count"]) if split == "train"
                     else len(row["window_indices"]))
            if count <= 0:
                raise ValueError(f"row has no windows: {row['clip_id']}")
            total += count
            self.cumulative.append(total)
        self[0]  # validate the cross-modal contract before workers start

    def __len__(self) -> int:
        return self.cumulative[-1]

    def row_and_window(self, index: int) -> tuple[dict, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row_index = bisect.bisect_right(self.cumulative, index)
        local = index - (self.cumulative[row_index - 1] if row_index else 0)
        row = self.rows[row_index]
        window = local if self.split == "train" else int(row["window_indices"][local])
        return row, window

    def __getitem__(self, index: int) -> dict:
        row, window = self.row_and_window(index)
        clip_id = str(row["clip_id"])

        path = sharded_path(self.latent_dir, clip_id, ".safetensors")
        with safe_open(path, framework="pt", device="cpu") as cached:
            if (cached.metadata() or {}).get(LATENT_LAYOUT[0]) != LATENT_LAYOUT[1]:
                raise ValueError(f"{path} lacks {LATENT_LAYOUT[0]}={LATENT_LAYOUT[1]}")
            latent = cached.get_slice("latents")[window]
            start = int(cached.get_slice("start_samples")[window])
            valid = int(cached.get_slice("valid_samples")[window])
        if valid != WINDOW_SAMPLES:
            raise ValueError(f"partial window in manifest: {clip_id}:{window}")
        if tuple(latent.shape) != (LATENT_DIM, LATENT_FRAMES):
            raise ValueError(f"bad latent shape for {clip_id}:{window}: {tuple(latent.shape)}")
        latent = latent.transpose(0, 1).float().contiguous()

        with np.load(sharded_path(self.text_dir, clip_id, ".npz"), allow_pickle=False) as text:
            text_features = torch.from_numpy(text["text_features"].astype(np.float32))
            text_features_c = torch.from_numpy(text["text_features_c"].astype(np.float32))

        item = {
            "clip_id": clip_id,
            "window_index": torch.tensor(window, dtype=torch.long),
            "latent": latent,
            "text_features": text_features,
            "text_features_c": text_features_c,
        }
        if self.teacher_dir is not None:
            with np.load(sharded_path(self.teacher_dir, clip_id, ".npz"),
                         allow_pickle=False) as cached:
                packed = cached["packed_roll"][window]
                teacher_start = int(cached["start_samples"][window])
            if teacher_start != start:
                raise ValueError(f"audio/MIDI window mismatch for {clip_id}:{window}")
            roll = np.unpackbits(packed, axis=-1, count=self.teacher_dim, bitorder="little")
            if roll.shape != (LATENT_FRAMES, self.teacher_dim):
                raise ValueError(f"bad roll shape for {clip_id}: {roll.shape}")
            item["zs"] = torch.from_numpy(roll.astype(np.float32))
        return item


def build_dataset(cfg, split: str) -> JamendoWindows:
    data = cfg.data
    root = Path(data.root)
    teacher = cfg.repa.teacher if cfg.repa.enabled else None
    if teacher is not None and teacher not in TEACHERS:
        raise ValueError(f"unknown teacher {teacher!r}; choose from {sorted(TEACHERS)}")
    return JamendoWindows(
        root / data.manifests / f"{split}.jsonl",
        split,
        latent_dir=root / data.latents,
        text_dir=root / data.text,
        teacher_dir=(root / data.teachers[teacher]) if teacher else None,
        teacher_slices=TEACHERS[teacher] if teacher else 1,
    )


class ShardedEvalSampler(Sampler[int]):
    """Shard indices across ranks without padding or duplicates."""

    def __init__(self, dataset: Dataset, rank: int, world_size: int):
        self.indices = list(range(rank, len(dataset), world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def build_loader(dataset: Dataset, batch_size: int, num_workers: int, *, train: bool,
                 rank: int, world_size: int, pin_memory: bool = True):
    sampler = (DistributedSampler(dataset, rank=rank, shuffle=True) if train
               else ShardedEvalSampler(dataset, rank, world_size))
    loader = DataLoader(dataset, batch_size, sampler=sampler, num_workers=num_workers,
                        drop_last=train, persistent_workers=num_workers > 0,
                        pin_memory=pin_memory)
    return sampler, loader
