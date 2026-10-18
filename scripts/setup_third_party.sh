#!/usr/bin/env bash
# Clone the third-party code MIDI-REPA runs from source, at the commits used for the paper:
#   stable-audio-3     SAME-L autoencoder (latent extraction and decoding)
#   basic-pitch-torch  Basic Pitch teacher (Table 1 only)
#   attm-eval          official ICME 2026 ATTM FAD / CLAP scripts
# and fetch the null-prompt features used for classifier-free guidance.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${MIDI_REPA_THIRD_PARTY:-third_party}
mkdir -p "$DEST"

clone() {  # name url commit
  if [ ! -d "$DEST/$1/.git" ]; then git clone --quiet "$2" "$DEST/$1"; fi
  git -C "$DEST/$1" -c advice.detachedHead=false checkout --quiet "$3"
  echo "$1 @ $(git -C "$DEST/$1" rev-parse --short HEAD)"
}
clone stable-audio-3    https://github.com/Stability-AI/stable-audio-3              fa5ee841dd49bae0fa361fac26904adc27fd400e
clone basic-pitch-torch https://github.com/gudgud96/basic-pitch-torch               19459054e0f621dfa7ed0824bc3fd264688d2436
clone attm-eval         https://github.com/ntu-musicailab/ICME26-ATTM-GC-Evaluation 38be180827efb2d5d6c7a9e269606d5eb2087ad6

# basic-pitch-torch calls scipy.signal.gaussian, which current SciPy only provides as
# scipy.signal.windows.gaussian. Same function; no transcription changes.
sed -i 's/scipy\.signal\.gaussian(/scipy.signal.windows.gaussian(/' \
  "$DEST/basic-pitch-torch/basic_pitch_torch/note_creation.py"

"${PYTHON:-python}" - <<'PY'
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

# Null-prompt T5 and CLAP features distributed with the ATTM baseline (MeanAudio).
out = Path("assets/empty_string")
expected = {
    "empty_string_t5.pth": "c7e9a3adce14701ceefb9d7bdd75fa5fcadad11002bab73fa793274d706b224f",
    "empty_string_clap_c.pth": "1d17a4ac85c1c438b90c0b181a6bbfb6dd0da43fda828e1beca08e87323cd46c",
}
for name, sha in expected.items():
    path = out / name
    if not path.exists():
        hf_hub_download(repo_id="AndreasXi/MeanAudio", filename=name, local_dir=out)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha:
        raise SystemExit(f"SHA-256 mismatch for {path}: {digest}")
    print(f"{path} ok")
PY
