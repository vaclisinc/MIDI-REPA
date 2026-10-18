"""Basic Pitch roll on the same window grid (Table 1 baseline teacher).

Basic Pitch (PyTorch port, onset 0.5 / frame 0.3) transcribes the vocal-removed audio;
a frame is active for a pitch when a note interval intersects the frame's time span.
Same packed layout and ``start_samples`` as the MuScriptor rolls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from midi_repa.data import (LATENT_FRAMES, PITCH_MAX, PITCH_MIN, PITCHES, WINDOW_SAMPLES,
                            read_jsonl, sharded_path)
from midi_repa.prep.rolls import save_packed
from midi_repa.third_party import add_to_path

SAMPLE_RATE = 44_100
FRAME_SAMPLES = WINDOW_SAMPLES / LATENT_FRAMES


def notes_to_roll(notes, windows: int) -> np.ndarray:
    starts = np.arange(windows, dtype=np.int64) * WINDOW_SAMPLES
    offsets = np.arange(LATENT_FRAMES, dtype=np.float64) * FRAME_SAMPLES
    frame_start = (starts[:, None].astype(np.float64) + offsets[None, :]) / SAMPLE_RATE
    frame_end = frame_start + FRAME_SAMPLES / SAMPLE_RATE
    roll = np.zeros((windows, LATENT_FRAMES, PITCHES), dtype=np.uint8)
    for start_time, end_time, pitch, _amplitude, _bends in notes:
        if not PITCH_MIN <= pitch <= PITCH_MAX:
            continue
        active = (frame_start < end_time) & (frame_end > start_time)
        if active.any():
            roll[:, :, pitch - PITCH_MIN][active] = 1
    return roll


def build(root: Path, audio_dir: Path, onset: float = 0.5, frame: float = 0.3,
          worker: int = 0, workers: int = 1) -> None:
    import torch

    repo = add_to_path("basic-pitch-torch")
    from basic_pitch_torch import note_creation
    from basic_pitch_torch.inference import run_inference
    from basic_pitch_torch.model import BasicPitchTorch

    model = BasicPitchTorch()
    model.load_state_dict(torch.load(repo / "assets/basic_pitch_pytorch_icassp_2022.pth",
                                     map_location="cpu"))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    rows = {}
    for split in ("train", "validation"):
        for row in read_jsonl(root / "manifests" / f"{split}.jsonl"):
            rows[row["clip_id"]] = int(row["full_window_count"])
    ids = sorted(rows)[worker::workers]
    for position, clip_id in enumerate(ids, 1):
        destination = sharded_path(root / "teacher/basic_pitch_roll", clip_id, ".npz")
        if destination.is_file():
            continue
        with torch.no_grad():
            output = run_inference(str(Path(audio_dir) / f"{clip_id}.mp3"), model)
        _, notes = note_creation.model_output_to_notes(output, onset_thresh=onset,
                                                       frame_thresh=frame)
        save_packed(destination, notes_to_roll(notes, rows[clip_id]))
        if position % 100 == 0:
            print(f"basic pitch {position}/{len(ids)}", flush=True)
