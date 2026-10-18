"""FluxAudio-S, the ICME 2026 ATTM baseline backbone, sized for SAME-L latents.

Adapted from the official baseline (github.com/ntu-musicailab/ICME26-ATTM-GC-FluxAudio,
itself built on MeanAudio and MMAudio). The blocks, initialisation and parameter names
are unchanged, so checkpoints trained with the original code load with ``strict=True``.
The only architectural change is the size: 256-channel SAME-L latents and 108 frames
per 10-second clip instead of the original 20 x 312 mel-VAE latents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from midi_repa.model.embeddings import TimestepEmbedder
from midi_repa.model.low_level import MLP, ChannelLastConv1d, ConvMLP
from midi_repa.model.rotary import compute_rope_rotations
from midi_repa.model.transformer_layers import FinalBlock, JointBlock, MMDitSingleBlock

log = logging.getLogger(__name__)


@dataclass
class PreprocessedConditions:
    text_f: torch.Tensor
    text_f_c: torch.Tensor


class FluxAudio(nn.Module):
    """Flux-style MMDiT: ``depth - fused_depth`` joint text/audio blocks, then fused blocks.

    Blocks are indexed globally as L0..L11: L0-L3 are the joint blocks, L4-L11 the fused
    blocks (see :func:`midi_repa.repa.resolve_block`).
    """

    def __init__(self,
                 *,
                 latent_dim: int,
                 text_dim: int,
                 text_c_dim: int,
                 hidden_dim: int,
                 depth: int,
                 fused_depth: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 latent_seq_len: int,
                 text_seq_len: int = 77,
                 latent_mean: Optional[torch.Tensor] = None,
                 latent_std: Optional[torch.Tensor] = None,
                 empty_string_feat: Optional[torch.Tensor] = None,
                 empty_string_feat_c: Optional[torch.Tensor] = None,
                 use_rope: bool = False) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self._latent_seq_len = latent_seq_len
        self._text_seq_len = text_seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.mm_depth = depth - fused_depth

        self.audio_input_proj = nn.Sequential(
            ChannelLastConv1d(latent_dim, hidden_dim, kernel_size=7, padding=3),
            nn.SELU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
        )
        self.text_input_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            MLP(hidden_dim, hidden_dim * 4),
        )
        self.text_cond_proj = nn.Sequential(
            nn.Linear(text_c_dim, hidden_dim),
            MLP(hidden_dim, hidden_dim * 4),
        )
        self.final_layer = FinalBlock(hidden_dim, latent_dim)
        self.t_embed = TimestepEmbedder(hidden_dim, frequency_embedding_size=256, max_period=10000)

        # The last joint block is pre-only: it updates the audio stream but not the text stream.
        self.joint_blocks = nn.ModuleList([
            JointBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio,
                       pre_only=(i == depth - fused_depth - 1))
            for i in range(depth - fused_depth)
        ])
        self.fused_blocks = nn.ModuleList([
            MMDitSingleBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=3, padding=1)
            for _ in range(fused_depth)
        ])

        if latent_mean is None:
            # Placeholders; the real statistics are loaded with the checkpoint.
            assert latent_std is None
            latent_mean = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
            latent_std = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
        else:
            assert latent_std is not None
            assert latent_mean.numel() == latent_dim, f'{latent_mean.numel()=} != {latent_dim=}'
        if empty_string_feat is None:
            empty_string_feat = torch.zeros((text_seq_len, text_dim))
        if empty_string_feat_c is None:
            empty_string_feat_c = torch.zeros((text_c_dim))
        assert empty_string_feat.shape[-1] == text_dim
        assert empty_string_feat_c.shape[-1] == text_c_dim

        self.latent_mean = nn.Parameter(latent_mean.view(1, 1, -1), requires_grad=False)
        self.latent_std = nn.Parameter(latent_std.view(1, 1, -1), requires_grad=False)
        self.empty_string_feat = nn.Parameter(empty_string_feat, requires_grad=False)
        self.empty_string_feat_c = nn.Parameter(empty_string_feat_c, requires_grad=False)

        self.initialize_weights()
        if self.use_rope:
            self.initialize_rotations()
        else:
            self.latent_rot = None
            self.text_rot = None

    def initialize_rotations(self):
        head_dim = self.hidden_dim // self.num_heads
        latent_rot = compute_rope_rotations(self._latent_seq_len, head_dim, 10000,
                                            freq_scaling=1.0, device=self.device)
        text_rot = compute_rope_rotations(self._text_seq_len, head_dim, 10000,
                                          freq_scaling=1.0, device=self.device)
        self.latent_rot = nn.Buffer(latent_rot, persistent=False)
        self.text_rot = nn.Buffer(text_rot, persistent=False)

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks.
        for block in self.joint_blocks:
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].bias, 0)
        for block in self.fused_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers.
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x.sub_(self.latent_mean).div_(self.latent_std)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.latent_std).add_(self.latent_mean)

    def preprocess_conditions(self, text_f: torch.Tensor,
                              text_f_c: torch.Tensor) -> PreprocessedConditions:
        """Projections that do not depend on the latent or timestep (cached over sampling)."""
        assert text_f.shape[1] == self._text_seq_len, f'{text_f.shape=} {self._text_seq_len=}'
        text_f_c = self.text_cond_proj(text_f_c)  # (B, D)
        text_f = self.text_input_proj(text_f)  # (B, 77, D)
        return PreprocessedConditions(text_f=text_f, text_f_c=text_f_c)

    def predict_flow(self, latent: torch.Tensor, t: torch.Tensor,
                     conditions: PreprocessedConditions) -> torch.Tensor:
        assert latent.shape[1] == self._latent_seq_len, f'{latent.shape=} {self._latent_seq_len=}'
        text_f = conditions.text_f
        text_f_c = conditions.text_f_c

        latent = self.audio_input_proj(latent)  # (B, N, D)
        global_c = self.t_embed(t).unsqueeze(1) + text_f_c.unsqueeze(1)  # (B, 1, D)
        extended_c = global_c

        for block in self.joint_blocks:
            latent, text_f = block(latent, text_f, global_c, extended_c, self.latent_rot,
                                   self.text_rot)
        for block in self.fused_blocks:
            latent = block(latent, extended_c, self.latent_rot)

        return self.final_layer(latent, extended_c)  # (B, N, latent_dim)

    def forward(self, latent: torch.Tensor, text_f: torch.Tensor, text_f_c: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """latent: (B, N, C); text_f: (B, 77, 1024); text_f_c: (B, 512); t: (B,)."""
        conditions = self.preprocess_conditions(text_f, text_f_c)
        return self.predict_flow(latent, t, conditions)

    def get_empty_string_sequence(self, bs: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.empty_string_feat.unsqueeze(0).expand(bs, -1, -1),
                self.empty_string_feat_c.unsqueeze(0).expand(bs, -1))

    def get_empty_conditions(self, bs: int) -> PreprocessedConditions:
        empty_string_feat, empty_string_feat_c = self.get_empty_string_sequence(1)
        conditions = self.preprocess_conditions(empty_string_feat, empty_string_feat_c)
        conditions.text_f = conditions.text_f.expand(bs, -1, -1)
        conditions.text_f_c = conditions.text_f_c.expand(bs, -1)
        return conditions

    def ode_wrapper(self, t: torch.Tensor, latent: torch.Tensor,
                    conditions: PreprocessedConditions,
                    empty_conditions: PreprocessedConditions,
                    cfg_strength: float) -> torch.Tensor:
        """Classifier-free-guided velocity: ``w * v(c) + (1 - w) * v(empty)``."""
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        if cfg_strength < 1.0:
            return self.predict_flow(latent, t, conditions)
        cond = self.predict_flow(latent, t, conditions)
        uncond = self.predict_flow(latent, t, empty_conditions)
        return cfg_strength * cond + (1 - cfg_strength) * uncond

    @property
    def device(self) -> torch.device:
        return self.latent_mean.device

    @property
    def latent_seq_len(self) -> int:
        return self._latent_seq_len


def fluxaudio_s_same(**kwargs) -> FluxAudio:
    """FluxAudio-S (121.6M parameters) on SAME-L latents: 256 channels, 108 frames per 10 s."""
    num_heads = 7
    return FluxAudio(latent_dim=256,
                     text_dim=1024,
                     hidden_dim=64 * num_heads,
                     depth=12,
                     fused_depth=8,
                     num_heads=num_heads,
                     latent_seq_len=108,
                     **kwargs)


MODELS = {"fluxaudio_s_same": fluxaudio_s_same}


def build_model(name: str, **kwargs) -> FluxAudio:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](**kwargs)
