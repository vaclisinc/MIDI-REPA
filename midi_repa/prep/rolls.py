"""Symbolic targets on the SAME-L window grid (paper Sec. 2.2.2).

Every complete 10-second window becomes a binary 108 x 88 roll (MIDI 21-108). A frame is
active for a pitch when a note interval intersects it. Windows start at multiples of
441,000 samples, exactly where the latents were cut, and a partial last window is dropped.

* MuScriptor cube: one slice per instrument family, ``[108, 7, 88]`` flattened class-major
  to ``[108, 616]``. Drum notes keep their General MIDI key on the shared pitch axis.
* MuScriptor roll: the OR over the seven slices, ``[108, 88]``.

Stored as ``np.packbits(..., bitorder="little")`` with the window ``start_samples``.
"""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from midi_repa.data import (LATENT_FRAMES, PITCH_MAX, PITCH_MIN, PITCHES, WINDOW_SAMPLES,
                            read_jsonl, sharded_path)

PALETTE_FILE = Path(__file__).resolve().parents[2] / "assets" / "instrument_palette.json"


def load_palette() -> tuple[list[str], dict[str, int]]:
    """Cube slice order and the MIDI track name -> slice mapping of JamendoMIDI."""
    remap = json.loads(PALETTE_FILE.read_text(encoding="utf-8"))
    palette = [str(name) for name in remap["palette"]]
    index = {name: i for i, name in enumerate(palette)}
    return palette, {str(k): index[v] for k, v in remap["name_to_class"].items()}


def midi_to_cube(midi_path: Path, windows: int) -> np.ndarray:
    """``[windows, 108, 7 * 88]`` uint8 roll for the first ``windows`` complete windows."""
    import pretty_midi

    palette, name_to_class = load_palette()
    drum_class = palette.index("drums")
    frame_seconds = 10.0 / LATENT_FRAMES
    roll = np.zeros((windows * LATENT_FRAMES, len(palette), PITCHES), dtype=np.uint8)
    for instrument in pretty_midi.PrettyMIDI(str(midi_path)).instruments:
        name = (instrument.name or "").strip()
        if instrument.is_drum:
            slot = drum_class
        elif name in name_to_class:
            slot = name_to_class[name]
        else:
            raise ValueError(f"{midi_path}: track name {name!r} is not in the palette")
        for note in instrument.notes:
            if not PITCH_MIN <= note.pitch <= PITCH_MAX:
                continue
            start = max(0, math.floor(note.start / frame_seconds))
            end = min(roll.shape[0], math.ceil(note.end / frame_seconds))
            if end > start:
                roll[start:end, slot, note.pitch - PITCH_MIN] = 1
    return roll.reshape(windows, LATENT_FRAMES, len(palette) * PITCHES)


def merge_cube(cube: np.ndarray) -> np.ndarray:
    """``[w, 108, 616]`` -> ``[w, 108, 88]``: a pitch is active if any instrument plays it."""
    windows = cube.shape[0]
    return cube.reshape(windows, LATENT_FRAMES, -1, PITCHES).any(axis=2).astype(np.uint8)


def save_packed(path: Path, roll: np.ndarray) -> None:
    starts = np.arange(roll.shape[0], dtype=np.int64) * WINDOW_SAMPLES
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, packed_roll=np.packbits(roll, axis=-1, bitorder="little"),
                 start_samples=starts)
    os.replace(temporary, path)


def _one(job: tuple[str, int, str, str, str]) -> int:
    clip_id, windows, midi_root, cube_root, roll_root = job
    cube_path = sharded_path(Path(cube_root), clip_id, ".npz")
    roll_path = sharded_path(Path(roll_root), clip_id, ".npz")
    if cube_path.is_file() and roll_path.is_file():
        return 0
    cube = midi_to_cube(sharded_path(Path(midi_root), clip_id, ".mid"), windows)
    save_packed(cube_path, cube)
    save_packed(roll_path, merge_cube(cube))
    return 1


def build(root: Path, workers: int = 16) -> None:
    """Cube and merged roll for every track in the train and validation manifests."""
    rows = {}
    for split in ("train", "validation"):
        for row in read_jsonl(root / "manifests" / f"{split}.jsonl"):
            rows[row["clip_id"]] = int(row["full_window_count"])
    jobs = [(clip_id, windows, str(root / "midi"), str(root / "teacher/muscriptor_cube"),
             str(root / "teacher/muscriptor_roll")) for clip_id, windows in sorted(rows.items())]
    created = 0
    with ProcessPoolExecutor(workers) as pool:
        for position, made in enumerate(pool.map(_one, jobs, chunksize=64), 1):
            created += made
            if position % 5000 == 0:
                print(f"rolls {position}/{len(jobs)}", flush=True)
    print(f"MuScriptor cube + roll: {created} written, {len(jobs) - created} already present")
