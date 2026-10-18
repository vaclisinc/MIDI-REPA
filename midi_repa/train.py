"""Train FluxAudio-S with (or without) MIDI-REPA.

    torchrun --nproc_per_node=1 -m midi_repa.train configs/arms/mus_roll_l2.yaml \
        data.root=/path/to/prepared output_dir=runs/mus_roll_l2

The paper trains every arm on one GPU with global batch 128 for 300k updates; with N
GPUs the batch is split evenly. Training resumes from ``<output_dir>/ckpt_last.pth``.
At the end the post-hoc EMA with ``sigma_rel = 0.05`` is written to ``ema_final.pth``,
the checkpoint used for all evaluations.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from midi_repa.config import load_config, repo_path
from midi_repa.data import build_dataset, build_loader
from midi_repa.ema import make_ema, synthesize_ema
from midi_repa.flow import FlowMatching, logit_normal_t
from midi_repa.model import build_model
from midi_repa.repa import BlockTap, RepaHead, RepaHeads, repa_loss

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger("midi_repa.train")


def load_empty_string_features(directory: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """T5 and CLAP features of the empty prompt, the null condition for CFG."""
    t5 = torch.load(directory / "empty_string_t5.pth", weights_only=True)[0]
    clap = torch.load(directory / "empty_string_clap_c.pth", weights_only=True)[0]
    return t5, clap


class Trainer:

    def __init__(self, cfg, rank: int, world_size: int, latent_mean: torch.Tensor,
                 latent_std: torch.Tensor):
        self.cfg = cfg
        self.rank = rank
        self.out = Path(cfg.output_dir)
        self.use_amp = bool(cfg.train.amp)
        self.wandb = None
        if cfg.wandb.enabled and rank == 0:
            import wandb
            self.wandb = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                                    name=cfg.run_name, id=cfg.run_name, resume="allow",
                                    config=OmegaConf.to_container(cfg, resolve=True))

        empty_t5, empty_clap = load_empty_string_features(repo_path(cfg.assets.empty_string))
        self.network = DDP(build_model(cfg.model,
                                       latent_mean=latent_mean,
                                       latent_std=latent_std,
                                       empty_string_feat=empty_t5,
                                       empty_string_feat_c=empty_clap,
                                       use_rope=cfg.use_rope,
                                       text_c_dim=512).cuda(),
                           device_ids=[rank], broadcast_buffers=False)
        self.fm = FlowMatching(0.0, num_steps=int(cfg.sampling.num_steps))
        self.ema = make_ema(self.network.module, cfg).cuda() if rank == 0 else None

        self.rng = torch.Generator(device="cuda")
        self.rng.manual_seed(int(cfg.seed) + rank)

        self.repa = bool(cfg.repa.enabled)
        self.repa_heads = None
        self.repa_taps: list[BlockTap] = []
        if self.repa:
            from midi_repa.data import PITCHES, TEACHERS
            target_dim = TEACHERS[cfg.repa.teacher] * PITCHES
            head = RepaHead(self.network.module.hidden_dim, target_dim)
            self.repa_heads = DDP(RepaHeads([head]).cuda(), device_ids=[rank],
                                  broadcast_buffers=False)
            self.repa_taps = [BlockTap(self.network, int(cfg.repa.layer))]
            self.repa_lambda = float(cfg.repa["lambda"])
            self.repa_t_min = float(cfg.repa.t_min)
            self.repa_t_max = float(cfg.repa.t_max)
            log.info("REPA: teacher=%s layer=L%d lambda=%s t in [%s, %s] head 448->%d",
                     cfg.repa.teacher, cfg.repa.layer, self.repa_lambda, self.repa_t_min,
                     self.repa_t_max, target_dim)

        train = cfg.train
        groups = [{"params": [p for p in self.network.parameters() if p.requires_grad],
                   "lr": train.learning_rate, "weight_decay": train.weight_decay}]
        if self.repa_heads is not None:
            groups.append({"params": list(self.repa_heads.parameters()),
                           "lr": train.learning_rate, "weight_decay": train.weight_decay})
        self.optimizer = optim.AdamW(groups, lr=train.learning_rate,
                                     weight_decay=train.weight_decay, betas=[0.9, 0.95],
                                     eps=1e-6 if self.use_amp else 1e-8, fused=True)
        self.scaler = torch.amp.GradScaler(init_scale=2048) if train.grad_scaler else None
        warmup = int(train.warmup_steps)
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            [optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: (step + 1) / (warmup + 1)),
             optim.lr_scheduler.MultiStepLR(self.optimizer, list(train.lr_milestones),
                                            train.lr_gamma)],
            [warmup])
        log.info("parameters: network %d, REPA head %d",
                 sum(p.numel() for p in self.network.parameters()),
                 sum(p.numel() for p in self.repa_heads.parameters()) if self.repa_heads else 0)

    # ----------------------------------------------------------------------------- losses

    def _flow_loss(self, data):
        """One flow-matching draw. The RNG call order matches the original implementation."""
        latent = data["latent"].cuda(non_blocking=True)
        text_f = data["text_features"].cuda(non_blocking=True)
        text_f_c = data["text_features_c"].cuda(non_blocking=True)
        # The original data path sampled x1 = mean + std * eps with std = 0 for these
        # deterministic latents. The draw is kept so that seeds reproduce the paper runs.
        eps = torch.empty_like(latent).normal_(generator=self.rng)
        x1 = latent + torch.zeros_like(latent) * eps
        x1 = self.network.module.normalize(x1)
        t = logit_normal_t(x1, generator=self.rng, m=self.cfg.train.t_sampling.mean,
                           s=self.cfg.train.t_sampling.scale)
        x0, xt = self.fm.get_x0_xt(x1, t, generator=self.rng)

        # Classifier-free guidance training: drop T5 and CLAP conditions together.
        null = torch.rand(x1.shape[0], device=x1.device, generator=self.rng) < \
            float(self.cfg.train.null_condition_probability)
        text_f[null] = self.network.module.empty_string_feat
        text_f_c[null] = self.network.module.empty_string_feat_c

        pred_v = self.network(xt, text_f, text_f_c, t)
        per_sample = self.fm.loss(pred_v, x0, x1)
        return per_sample.mean(), t

    def _repa_loss(self, zs: torch.Tensor, t: torch.Tensor):
        """Eq. 1 on the batch elements whose t lies in the alignment interval."""
        preds = self.repa_heads([tap.latest for tap in self.repa_taps])
        in_window = (t >= self.repa_t_min) & (t <= self.repa_t_max)
        if not in_window.any():
            return sum(p.sum() * 0.0 for p in preds), None
        loss, frac_valid = repa_loss(preds[0][in_window], zs[in_window])
        return loss, frac_valid

    def _total_loss(self, data):
        for tap in self.repa_taps:
            tap.clear()
        flow_loss, t = self._flow_loss(data)
        total = flow_loss
        proj_loss = None
        if self.repa:
            proj_loss, _ = self._repa_loss(data["zs"].cuda(non_blocking=True), t)
            total = total + self.repa_lambda * proj_loss
        return total, flow_loss, proj_loss

    # ------------------------------------------------------------------------- steps

    def _params_with_grad(self) -> list[torch.Tensor]:
        return [p for group in self.optimizer.param_groups for p in group["params"]
                if p.grad is not None]

    def train_step(self, data) -> dict:
        self.network.train()
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.bfloat16):
            total, flow_loss, proj_loss = self._total_loss(data)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss: flow={float(flow_loss)}")

        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self._params_with_grad(),
                                                       self.cfg.train.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self._params_with_grad(),
                                                       self.cfg.train.clip_grad_norm)
            self.optimizer.step()

        # The tap holds a non-leaf activation; clear it before the EMA deep-copies hooks.
        for tap in self.repa_taps:
            tap.clear()
        if self.ema is not None:
            self.ema.update()
        self.scheduler.step()
        self.network.eval()
        out = {"train/flow": flow_loss.detach().float(), "train/total": total.detach().float(),
               "grad_norm": grad_norm.detach().float()}
        if proj_loss is not None:
            out["train/repa"] = proj_loss.detach().float()
        return out

    @torch.inference_mode()
    def validate(self, loader) -> dict:
        """Loss on the 444 held-out monitoring windows; never used for selection."""
        self.network.eval()
        sums = {"flow": 0.0, "repa": 0.0}
        batches = 0
        for data in loader:
            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.bfloat16):
                _, flow_loss, proj_loss = self._total_loss(data)
            sums["flow"] += float(flow_loss)
            sums["repa"] += float(proj_loss) if proj_loss is not None else 0.0
            batches += 1
        values = torch.tensor([sums["flow"], sums["repa"], batches], device="cuda",
                              dtype=torch.float64)
        dist.all_reduce(values)
        metrics = {"val/flow": float(values[0] / values[2])}
        if self.repa:
            metrics["val/repa"] = float(values[1] / values[2])
        return metrics

    # -------------------------------------------------------------------- checkpoints

    def save_checkpoint(self, it: int, copy: bool = False) -> None:
        if self.rank != 0:
            return
        self.out.mkdir(parents=True, exist_ok=True)
        state = {
            "it": it,
            "weights": self.network.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "repa_heads": self.repa_heads.module.state_dict() if self.repa_heads else None,
        }
        if copy:
            torch.save(state, self.out / f"ckpt_{it}.pth")
        last = self.out / "ckpt_last.pth"
        if last.exists():
            last.replace(self.out / "ckpt_shadow.pth")
        torch.save(state, last)
        log.info("checkpoint %d -> %s", it, last)

    def load_checkpoint(self, path: Path) -> int:
        state = torch.load(path, map_location=f"cuda:{self.rank}", weights_only=True)
        self.network.module.load_state_dict(state["weights"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if self.ema is not None:
            self.ema.load_state_dict(state["ema"])
        if self.repa_heads is not None and state.get("repa_heads") is not None:
            self.repa_heads.module.load_state_dict(state["repa_heads"])
        log.info("resumed from %s at iteration %d", path, state["it"])
        return int(state["it"])


def train(cfg) -> None:
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    world_size = dist.get_world_size()
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    out = Path(cfg.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, out / "config.yaml")

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    random.seed(int(cfg.seed))

    batch_size = int(cfg.train.batch_size) // world_size
    train_set = build_dataset(cfg, "train")
    val_set = build_dataset(cfg, "validation")
    sampler, loader = build_loader(train_set, batch_size, int(cfg.data.num_workers), train=True,
                                   rank=rank, world_size=world_size,
                                   pin_memory=bool(cfg.data.pin_memory))
    _, val_loader = build_loader(val_set, batch_size, int(cfg.data.num_workers), train=False,
                                 rank=rank, world_size=world_size,
                                 pin_memory=bool(cfg.data.pin_memory))
    log.info("train windows %d (%d batches/epoch), validation windows %d",
             len(train_set), len(loader), len(val_set))

    stats = repo_path(cfg.assets.latent_stats)
    latent_mean = torch.load(stats / "latent_mean.pt")
    latent_std = torch.load(stats / "latent_std.pt")
    trainer = Trainer(cfg, rank, world_size, latent_mean, latent_std)
    eval_rng = trainer.rng.graphsafe_get_state()

    it = trainer.load_checkpoint(out / "ckpt_last.pth") if (out / "ckpt_last.pth").exists() else 0
    total = int(cfg.train.num_iterations)
    epoch = it // len(loader)
    log.info("training %d -> %d (~%d epochs)", it, total, math.ceil(total / len(loader)) - epoch)

    try:
        while it < total:
            sampler.set_epoch(epoch)
            epoch += 1
            for data in loader:
                metrics = trainer.train_step(data)
                step = it + 1
                if step % int(cfg.train.log_interval) == 0 and rank == 0:
                    lr = trainer.scheduler.get_last_lr()[0]
                    log.info("it %d lr %.2e %s", step, lr,
                             " ".join(f"{k}={float(v):.4f}" for k, v in metrics.items()))
                    if trainer.wandb is not None:
                        trainer.wandb.log({"lr": lr, **{k: float(v) for k, v in metrics.items()}},
                                          step=step)
                if step % int(cfg.train.val_interval) == 0:
                    # Every validation pass sees the same noise.
                    train_rng = trainer.rng.graphsafe_get_state()
                    trainer.rng.graphsafe_set_state(eval_rng)
                    val = trainer.validate(val_loader)
                    trainer.rng.graphsafe_set_state(train_rng)
                    if rank == 0:
                        log.info("it %d %s", step, val)
                        if trainer.wandb is not None:
                            trainer.wandb.log(val, step=step)
                if step % int(cfg.train.checkpoint_interval) == 0 or step == total:
                    trainer.save_checkpoint(step, copy=(step == total))
                it = step
                if it >= total:
                    break
    finally:
        if it < total:
            trainer.save_checkpoint(it)

    dist.barrier()
    del trainer
    torch.cuda.empty_cache()
    if rank == 0:
        sigma = float(cfg.ema.output_sigma)
        torch.save(synthesize_ema(cfg, sigma), out / "ema_final.pth")
        log.info("EMA sigma_rel=%s -> %s", sigma, out / "ema_final.pth")
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="arm config, e.g. configs/arms/mus_roll_l2.yaml")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    args = parser.parse_args()
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
