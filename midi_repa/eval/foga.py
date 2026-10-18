"""Forward-only Gate Ablation (FoG-A, AG-REPA) of the baseline (paper Table 2).

For block ``k``, its residual contribution is removed (the block returns its input) and
the change of the predicted velocity is measured:

    FoG-A_k(t) = E[ ||v^{-k}(x_t, t, c) - v(x_t, t, c)||^2 / (||v(x_t, t, c)||^2 + eps) ]

over 512 training windows at t in {0, 0.25, 0.5, 0.75, 1}; Table 2 reports the mean over t.
The paper used the final raw (non-EMA) baseline weights, ``ckpt_300000.pth``.

    python -m midi_repa.eval.foga --checkpoint runs/baseline/ckpt_300000.pth \
        --config configs/arms/baseline.yaml data.root=/path/to/prepared --out foga.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from midi_repa.config import load_config
from midi_repa.data import build_dataset
from midi_repa.generate import load_generator
from midi_repa.repa import BlockAblate


@torch.no_grad()
def foga_at_t(net, ablators, dataset, indices, t_value: float, seed: int, batch_size: int,
              eps: float) -> dict[int, float]:
    device = net.device
    total = {layer: 0.0 for layer in ablators}
    # Per-t noise, shared by the unablated and every ablated forward of a batch.
    generator = torch.Generator(device=device).manual_seed(seed * 100003 + int(round(t_value * 1000)))
    for start in range(0, len(indices), batch_size):
        items = [dataset[i] for i in indices[start:start + batch_size]]
        x1 = net.normalize(torch.stack([it["latent"] for it in items]).to(device))
        text_f = torch.stack([it["text_features"] for it in items]).to(device)
        text_f_c = torch.stack([it["text_features_c"] for it in items]).to(device)
        x0 = torch.randn(x1.shape, generator=generator, device=device, dtype=x1.dtype)
        xt = (1.0 - t_value) * x1 + t_value * x0
        t = torch.full((x1.shape[0],), float(t_value), device=device)
        v = net(xt, text_f, text_f_c, t)
        v_sq = v.pow(2).flatten(1).sum(1)
        for layer, ablator in ablators.items():
            ablator.enabled = True
            v_k = net(xt, text_f, text_f_c, t)
            ablator.enabled = False
            total[layer] += ((v_k - v).pow(2).flatten(1).sum(1) / (v_sq + eps)).sum().item()
    return {layer: value / len(indices) for layer, value in total.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/arms/baseline.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-windows", type=int, default=512)
    parser.add_argument("--ts", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. data.root=/data")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    cfg.repa.enabled = False  # FoG-A never reads a symbolic target
    dataset = build_dataset(cfg, "train")
    # Consecutive indices are windows of one track, so sample windows across the corpus.
    indices = sorted(int(i) for i in
                     np.random.default_rng(args.seed).permutation(len(dataset))[:args.n_windows])

    net = load_generator(args.checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    blocks = len(net.joint_blocks) + len(net.fused_blocks)
    ablators = {layer: BlockAblate(net, layer) for layer in range(blocks)}
    per_t = {}
    for t_value in args.ts:
        per_t[str(float(t_value))] = {str(k): v for k, v in foga_at_t(
            net, ablators, dataset, indices, t_value, args.seed, args.batch_size, args.eps).items()}
        print(f"t={t_value}: " + " ".join(f"L{k}={v:.4f}" for k, v in per_t[str(float(t_value))].items()),
              flush=True)
    mean = {str(k): float(np.mean([per_t[str(float(t))][str(k)] for t in args.ts]))
            for k in range(blocks)}
    top3 = sorted(range(blocks), key=lambda k: -mean[str(k)])[:3]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "n_windows": len(indices), "ts": args.ts,
        "per_t": per_t, "t_avg": mean, "top3": top3,
        "windows": [{"index": i, "clip_id": dataset.row_and_window(i)[0]["clip_id"]}
                    for i in indices]}, indent=2) + "\n")
    print("t-averaged FoG-A: " + " ".join(f"L{k}={mean[str(k)]:.4f}" for k in range(blocks)))
    print(f"top-3 blocks: {['L%d' % k for k in top3]} -> {args.out}")


if __name__ == "__main__":
    main()
