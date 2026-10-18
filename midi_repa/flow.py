"""Conditional flow matching with the reverse-time convention of the FluxAudio baseline.

``x_t = (1 - t) * x_data + t * x_noise``, so ``t = 1`` is pure noise and ``t = 0`` is music,
and the target velocity is ``x_noise - x_data``. Sampling integrates from ``t = 1`` to ``t = 0``.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


def logit_normal_t(x: torch.Tensor, generator: Optional[torch.Generator] = None,
                   m: float = 0.0, s: float = 1.0) -> torch.Tensor:
    """``t = sigmoid(u)``, ``u ~ N(m, s^2)``, one per batch element."""
    u = torch.randn(x.shape[0], device=x.device, generator=generator) * s + m
    return torch.sigmoid(u)


class FlowMatching:

    def __init__(self, min_sigma: float = 0.0, inference_mode: str = 'euler',
                 num_steps: int = 25):
        if inference_mode != 'euler':
            raise ValueError("only the Euler sampler used in the paper is provided")
        self.min_sigma = min_sigma
        self.inference_mode = inference_mode
        self.num_steps = num_steps

    @staticmethod
    def get_conditional_flow(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x0 is noise, x1 is data."""
        t = t[:, None, None].expand_as(x0)
        return (1 - t) * x1 + t * x0

    @staticmethod
    def loss(predicted_v: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared error against ``v = x0 - x1``."""
        reduce_dim = list(range(1, len(predicted_v.shape)))
        return (predicted_v - (x0 - x1)).pow(2).mean(dim=reduce_dim)

    def get_x0_xt(self, x1: torch.Tensor, t: torch.Tensor,
                  generator: Optional[torch.Generator] = None):
        x0 = torch.empty_like(x1).normal_(generator=generator)
        return x0, self.get_conditional_flow(x0, x1, t)

    def to_data(self, fn: Callable, x0: torch.Tensor) -> torch.Tensor:
        """Euler-integrate ``fn(t, x)`` from noise (t=1) to data (t=0)."""
        x = x0
        steps = torch.linspace(1, 0 - self.min_sigma, self.num_steps + 1)
        for ti, t in enumerate(steps[:-1]):
            flow = fn(t, x)
            x = x + (steps[ti + 1] - t) * flow
        return x
