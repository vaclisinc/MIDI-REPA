"""MIDI-REPA: align an intermediate DiT block with a symbolic (piano-roll) target.

Three pieces (paper Sec. 2.1, Eq. 1-2):

* :class:`BlockTap` captures the hidden state ``h_{l,t}`` of block ``l`` with a forward hook,
  so the backbone itself is untouched.
* :class:`RepaHead` is the projector ``g_phi``: one temporal ``Conv1d`` (kernel 3, padding 1)
  from the 448-dim hidden state to the target dimension (88 for a roll, 7 x 88 = 616 for the
  flattened instrument cube), in the spirit of iREPA.
* :func:`repa_loss` is the negative cosine similarity between projection and target,
  averaged over frames whose target is not all-zero.

Training applies the loss only to the batch elements whose timestep lies in the
alignment interval ``[t_min, t_max]`` (``L = L_flow + lambda * 1[t in T] * L_REPA``).
Neither the tap nor the head is used at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def unwrap(network: nn.Module) -> nn.Module:
    return network.module if hasattr(network, "module") else network


def resolve_block(network: nn.Module, global_index: int) -> nn.Module:
    """Map a global block index L0..L11 (joint blocks first, then fused) to its module."""
    net = unwrap(network)
    n_joint = len(net.joint_blocks)
    total = n_joint + len(net.fused_blocks)
    if not 0 <= global_index < total:
        raise IndexError(f"block index {global_index} out of range [0, {total})")
    if global_index < n_joint:
        return net.joint_blocks[global_index]
    return net.fused_blocks[global_index - n_joint]


class BlockTap:
    """Forward hook that keeps the latest audio-stream output of one block (grad attached)."""

    def __init__(self, network: nn.Module, global_index: int):
        self.module = resolve_block(network, global_index)
        self.global_index = global_index
        self.latest: torch.Tensor | None = None
        self._handle = self.module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        # A joint block returns (latent, text); a fused block returns latent.
        self.latest = output[0] if isinstance(output, (tuple, list)) else output

    def clear(self):
        self.latest = None

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class BlockAblate:
    """Forward hook that turns one block into the identity when ``enabled`` (FoG-A, Table 3).

    Returning the block's input in place of its output removes its residual contribution;
    for a joint block both the audio and the text streams pass through unchanged.
    """

    def __init__(self, network: nn.Module, global_index: int):
        self.module = resolve_block(network, global_index)
        self.global_index = global_index
        self.enabled = False
        self._handle = self.module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not self.enabled:
            return None
        if isinstance(output, (tuple, list)):
            in_text = inputs[1] if len(inputs) > 1 else output[1]
            return (inputs[0], in_text) + tuple(output[2:])
        return inputs[0]

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class RepaHead(nn.Module):
    """``g_phi``: a single temporal convolution, ``(B, T, hidden) -> (B, T, target_dim)``."""

    def __init__(self, hidden_size: int, target_dim: int, kernel: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.target_dim = target_dim
        self.net = nn.Sequential(nn.Conv1d(hidden_size, target_dim, kernel, padding=kernel // 2))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        assert h.shape[-1] == self.hidden_size, f"expected hidden {self.hidden_size}, got {h.shape}"
        return self.net(h.transpose(1, 2)).transpose(1, 2)


class RepaHeads(nn.Module):
    """Container with a real ``forward`` so DDP synchronises the head's gradients.

    The ``heads.<i>`` layout matches the auxiliary-module checkpoints of the original runs.
    """

    def __init__(self, heads: list[nn.Module]):
        super().__init__()
        self.heads = nn.ModuleList(heads)

    def forward(self, hidden_list: list[torch.Tensor]) -> list[torch.Tensor]:
        return [head(x) for head, x in zip(self.heads, hidden_list)]


def repa_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked negative cosine similarity (Eq. 1). Returns ``(loss, fraction_of_valid_frames)``.

    ``pred`` and ``target`` are ``(B, T, D)``. A frame is valid when its target has any
    active cell; silent frames carry no symbolic content and are excluded from the mean.
    The mask is cast to ``pred``'s dtype, as in the original runs (bf16 under autocast).
    """
    assert pred.shape == target.shape, f"pred {pred.shape} vs target {target.shape}"
    valid = (target.amax(dim=-1) > 0.0).to(pred.dtype)
    cos = (F.normalize(pred, dim=-1) * F.normalize(target, dim=-1)).sum(dim=-1)
    loss = -(cos * valid).sum() / valid.sum().clamp_min(1.0)
    return loss, valid.mean()
