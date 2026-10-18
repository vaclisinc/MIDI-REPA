"""Frozen SAME-L autoencoder (Stable Audio 3): 44.1 kHz stereo <-> 256 x 108 latents per 10 s.

The paper used the autoencoder bundled with ``stabilityai/stable-audio-3-medium`` for both
encoding and decoding. Its encoder is identical to the standalone ``stabilityai/SAME-L``
release, but its decoder weights differ, so decode with the bundled one to reproduce the
paper's audio and scores (the Hugging Face repository asks you to accept its license).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from midi_repa.third_party import add_to_path

RELEASE = "stabilityai/stable-audio-3-medium"


def load_same_l(device: str, config: str | Path | None = None,
                checkpoint: str | Path | None = None):
    """Load the SAME-L autoencoder from local files, or from the stable-audio-3-medium release."""
    add_to_path("stable-audio-3")
    from stable_audio_3 import AutoencoderModel
    from stable_audio_3.loading_utils import load_autoencoder

    if (config is None) != (checkpoint is None):
        raise ValueError("pass both config and checkpoint, or neither")
    if config is None:
        from huggingface_hub import hf_hub_download
        config = hf_hub_download(RELEASE, "model_config.json")
        checkpoint = hf_hub_download(RELEASE, "model.safetensors")

    sample_rate = int(json.loads(Path(config).read_text(encoding="utf-8"))["sample_rate"])
    autoencoder = load_autoencoder(str(config), str(checkpoint), device=device)
    autoencoder.eval().requires_grad_(False)
    return AutoencoderModel(autoencoder, sample_rate, device)


@torch.no_grad()
def decode(autoencoder, latents: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
    """Unnormalised latents ``(B, 108, 256)`` -> stereo audio ``(B, 2, samples)`` in fp32."""
    batch_size = batch_size or len(latents)
    out = []
    for begin in range(0, len(latents), batch_size):
        batch = latents[begin:begin + batch_size]
        out.append(autoencoder.decode(batch.transpose(1, 2).contiguous().float()).cpu())
    return torch.cat(out)
