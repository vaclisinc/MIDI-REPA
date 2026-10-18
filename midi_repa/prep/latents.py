"""SAME-L latents of non-overlapping 10-second windows, one safetensors file per track.

Audio is resampled to 44.1 kHz stereo and cut at multiples of 441,000 samples; the last
window is zero-padded and its true length recorded in ``valid_samples`` (only complete
windows are trained on). Each window is encoded as one context (``chunked=False``) in fp16.

The SAME-L encoder is stochastic: stable-audio-3 perturbs its learned query tokens with
Gaussian noise (``mask_noise``) even in eval mode. Here the noise is seeded per track
(``SEED + track_id``) so a rebuild is reproducible. The paper's cache drew this noise from
one RNG stream per encoding worker, so recomputed latents differ from it by about 1% of the
mean latent magnitude (the same as two seeds differ from each other).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from midi_repa.autoencoder import load_same_l
from midi_repa.data import LATENT_DIM, LATENT_FRAMES, LATENT_LAYOUT, WINDOW_SAMPLES, sharded_path
from midi_repa.prep.jamendomidi import read_splits

SAMPLE_RATE = 44_100
SEED = 20260803


def _complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with safe_open(path, framework="np") as handle:
            return (handle.metadata() or {}).get(LATENT_LAYOUT[0]) == LATENT_LAYOUT[1]
    except Exception:
        return False


def encode_track(model, audio_path: Path, destination: Path, batch_size: int = 8) -> int:
    import torchaudio

    audio, rate = torchaudio.load(str(audio_path))
    audio = audio.repeat(2, 1) if audio.shape[0] == 1 else audio[:2]
    if rate != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, rate, SAMPLE_RATE)
    total = int(audio.shape[-1])
    count = math.ceil(total / WINDOW_SAMPLES)
    latents, valid, starts = [], [], []
    for first in range(0, count, batch_size):
        windows = []
        for index in range(first, min(first + batch_size, count)):
            start = index * WINDOW_SAMPLES
            window = audio[:, start:start + WINDOW_SAMPLES]
            valid.append(int(window.shape[-1]))
            starts.append(start)
            if window.shape[-1] < WINDOW_SAMPLES:
                window = torch.nn.functional.pad(window, (0, WINDOW_SAMPLES - window.shape[-1]))
            windows.append(window)
        batch = torch.stack(windows).to(model.device, dtype=torch.float16)
        encoded = model.encode(batch, SAMPLE_RATE, chunked=False)
        if tuple(encoded.shape[1:]) != (LATENT_DIM, LATENT_FRAMES):
            raise ValueError(f"unexpected latent shape {tuple(encoded.shape)}")
        # The encoder returns a channel-last view; safetensors stores raw bytes, so force
        # C order or the file is silently transposed.
        latents.append(np.ascontiguousarray(encoded.cpu().numpy(), dtype=np.float16))
    tensors = {"latents": np.ascontiguousarray(np.concatenate(latents)),
               "start_samples": np.asarray(starts, dtype=np.int64),
               "valid_samples": np.asarray(valid, dtype=np.int32)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    save_file(tensors, temporary, metadata={LATENT_LAYOUT[0]: LATENT_LAYOUT[1],
                                            "sample_rate": str(SAMPLE_RATE),
                                            "window_samples": str(WINDOW_SAMPLES)})
    written = load_file(temporary)
    if any(not np.array_equal(written[k], v) for k, v in tensors.items()):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"round-trip mismatch writing {destination}")
    os.replace(temporary, destination)
    return count


def build(root: Path, audio_dir: Path, same_config: Path | None = None,
          same_checkpoint: Path | None = None, worker: int = 0, workers: int = 1,
          batch_size: int = 8) -> None:
    """Encode every train/validation track; ``worker``/``workers`` shard the track list."""
    ids = [r["track_id"] for r in read_splits(root) if r["split"] in ("train", "validation")]
    ids = ids[worker::workers]
    model = load_same_l("cuda", same_config, same_checkpoint)
    model.autoencoder.to(torch.float16)
    done = 0
    for position, clip_id in enumerate(ids, 1):
        destination = sharded_path(root / "latents", clip_id, ".safetensors")
        if not _complete(destination):
            torch.manual_seed(SEED + int(clip_id))
            with torch.inference_mode():
                encode_track(model, Path(audio_dir) / f"{clip_id}.mp3", destination, batch_size)
            done += 1
        if position % 100 == 0:
            print(f"latents {position}/{len(ids)} (encoded {done})", flush=True)
    print(f"latents: encoded {done}, {len(ids) - done} already present")
