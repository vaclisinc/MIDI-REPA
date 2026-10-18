"""Locate third-party code cloned by ``scripts/setup_third_party.sh``.

stable-audio-3 (SAME-L), basic-pitch-torch and the official ATTM evaluation are used from
pinned source checkouts rather than installed: stable-audio-3 declares ``transformers>=5``
while the T5/CLAP text stack of the baseline requires ``transformers<5``, and the SAME-L
autoencoder path runs fine on the latter.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MIDI_REPA_THIRD_PARTY", REPO / "third_party"))

PINS = {
    "stable-audio-3": ("https://github.com/Stability-AI/stable-audio-3",
                       "fa5ee841dd49bae0fa361fac26904adc27fd400e"),
    "basic-pitch-torch": ("https://github.com/gudgud96/basic-pitch-torch", "19459054e0f621dfa7ed0824bc3fd264688d2436"),
    "attm-eval": ("https://github.com/ntu-musicailab/ICME26-ATTM-GC-Evaluation", "38be180827efb2d5d6c7a9e269606d5eb2087ad6"),
}


def path(name: str) -> Path:
    location = ROOT / name
    if not location.is_dir():
        raise FileNotFoundError(
            f"{location} is missing; run scripts/setup_third_party.sh "
            f"(or set MIDI_REPA_THIRD_PARTY)")
    return location


def add_to_path(name: str) -> Path:
    location = path(name)
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))
    return location
