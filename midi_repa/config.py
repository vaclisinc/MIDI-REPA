"""Experiment configs: ``configs/base.yaml`` merged with one arm file and CLI overrides."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "configs" / "base.yaml"


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> DictConfig:
    """``base.yaml`` <- ``path`` <- ``overrides`` (OmegaConf dotlist, e.g. ``data.root=/x``)."""
    cfg = OmegaConf.load(BASE)
    if path is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg


def repo_path(value: str | Path) -> Path:
    """Resolve a path given relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else REPO / path
