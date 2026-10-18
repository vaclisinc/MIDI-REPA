"""Text conditioning of FluxAudio-S: FLAN-T5-large token features (77 x 1024) and a
LAION-CLAP text embedding (512) from the music/speech/AudioSet checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

T5_MODEL = "google/flan-t5-large"
CLAP_REPO = "lukewys/laion_clap"
CLAP_CHECKPOINT = "music_speech_audioset_epoch_15_esc_89.98.pt"
MAX_LENGTH = 77


def clap_checkpoint(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(CLAP_REPO, CLAP_CHECKPOINT))


class TextEncoder:

    def __init__(self, device: str, clap_ckpt: str | Path | None = None,
                 dtype: torch.dtype = torch.float32):
        from transformers import AutoTokenizer, T5EncoderModel
        import laion_clap

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
        self.t5 = T5EncoderModel.from_pretrained(T5_MODEL, torch_dtype=dtype)
        self.t5 = self.t5.eval().requires_grad_(False).to(device)
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base",
                                           device=str(device))
        self.clap.load_ckpt(str(clap_checkpoint(clap_ckpt)), verbose=False)
        self.clap.eval()

    @torch.no_grad()
    def __call__(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(prompts, max_length=MAX_LENGTH, padding="max_length",
                                truncation=True, return_tensors="pt")
        t5 = self.t5(input_ids=tokens.input_ids.to(self.device),
                     attention_mask=tokens.attention_mask.to(self.device))[0].float()
        clap = self.clap.get_text_embedding(prompts, use_tensor=True).float().to(self.device)
        return t5, clap
