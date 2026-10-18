"""Post-hoc EMA (Karras et al., 2024) via ``nitrous_ema``: track two profiles during
training and synthesise the evaluated ``sigma_rel = 0.05`` model at the end."""

from __future__ import annotations

from pathlib import Path

from nitrous_ema import PostHocEMA

from midi_repa.model import build_model


def make_ema(network, cfg) -> PostHocEMA:
    return PostHocEMA(network,
                      sigma_rels=list(cfg.ema.sigma_rels),
                      update_every=1,
                      checkpoint_every_num_steps=int(cfg.ema.checkpoint_every),
                      checkpoint_folder=str(Path(cfg.output_dir) / "ema_ckpts"),
                      step_size_correction=True)


def synthesize_ema(cfg, sigma_rel: float, step: int | None = None) -> dict:
    """Return the state dict of the EMA model with ``sigma_rel`` (at ``step``, default last)."""
    shell = build_model(cfg.model, text_c_dim=512)
    ema = PostHocEMA(shell,
                     sigma_rels=list(cfg.ema.sigma_rels),
                     update_every=1,
                     checkpoint_every_num_steps=int(cfg.ema.checkpoint_every),
                     checkpoint_folder=str(Path(cfg.output_dir) / "ema_ckpts"))
    model = ema.synthesize_ema_model(sigma_rel=sigma_rel, step=step, device="cpu")
    return model.ema_model.state_dict()
