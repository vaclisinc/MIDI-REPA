"""Generate 10-second clips from text prompts.

    python -m midi_repa.generate --checkpoint runs/mus_roll_l2/ema_final.pth \
        --prompts assets/eval/sdd_prompts.txt --seed 42 --out samples/mus_roll_l2/sdd/seed42

Writes ``gen_<i>.wav`` (44.1 kHz stereo) for line ``i`` of the prompt file, plus
``meta/prompts.txt``. The initial noise for all prompts is drawn at once from ``--seed``,
so row ``i`` of the prior is the same for every checkpoint: arms compared at the same
seed see identical noise. Evaluation settings of the paper: post-hoc EMA
``sigma_rel = 0.05``, 25 Euler steps, CFG 4.5, generation batch 10.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import soundfile as sf
import torch

from midi_repa.autoencoder import decode, load_same_l
from midi_repa.flow import FlowMatching
from midi_repa.model import FluxAudio, build_model
from midi_repa.text import TextEncoder

SAMPLE_RATE = 44_100


def load_generator(checkpoint: Path, device: str, model: str = "fluxaudio_s_same",
                   use_rope: bool = True) -> FluxAudio:
    """Load a FluxAudio checkpoint: a plain state dict (``ema_final.pth``) or ``{"weights": ...}``.

    The checkpoint carries the latent statistics and null-prompt features as parameters.
    """
    net = build_model(model, use_rope=use_rope, text_c_dim=512)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "weights" in state:
        state = state["weights"]
    net.load_state_dict(state)  # strict: also restores latent statistics and null prompts
    return net.to(device).eval().requires_grad_(False)


@torch.no_grad()
def sample_latents(net: FluxAudio, text_f: torch.Tensor, text_f_c: torch.Tensor, seed: int,
                   cfg_strength: float = 4.5, num_steps: int = 25,
                   batch_size: int = 10) -> torch.Tensor:
    """Unnormalised SAME-L latents ``(n, 108, 256)`` for ``n`` prompts."""
    device = text_f.device
    n = len(text_f)
    fm = FlowMatching(num_steps=num_steps)
    rng = torch.Generator(device=device).manual_seed(seed)
    prior = torch.empty(n, net.latent_seq_len, net.latent_dim, device=device).normal_(
        generator=rng)
    batches = []
    for begin in range(0, n, batch_size):
        end = min(begin + batch_size, n)
        cond = net.preprocess_conditions(text_f[begin:end], text_f_c[begin:end])
        empty = net.get_empty_conditions(end - begin)
        batches.append(fm.to_data(
            lambda t, x: net.ode_wrapper(t, x, cond, empty, cfg_strength), prior[begin:end]))
    return net.unnormalize(torch.cat(batches).clone())


def write_wavs(audio: torch.Tensor, out: Path, prefix: str = "gen") -> None:
    out.mkdir(parents=True, exist_ok=True)
    for i, wav in enumerate(audio):
        sf.write(str(out / f"{prefix}_{i}.wav"), wav.cpu().transpose(0, 1).numpy(), SAMPLE_RATE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="one prompt per line")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=None, help="first n prompts (default: all)")
    parser.add_argument("--cfg-strength", type=float, default=4.5)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--same-config", type=Path, default=None)
    parser.add_argument("--same-checkpoint", type=Path, default=None)
    parser.add_argument("--clap-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = [line for line in args.prompts.read_text(encoding="utf-8").splitlines() if line]
    prompts = prompts[:args.n] if args.n is not None else prompts

    encoder = TextEncoder(device, args.clap_checkpoint)
    text_f, text_f_c = encoder(prompts)
    del encoder
    net = load_generator(args.checkpoint, device)
    latents = sample_latents(net, text_f, text_f_c, args.seed, args.cfg_strength,
                             args.num_steps, args.batch_size)
    del net, text_f, text_f_c
    gc.collect()
    torch.cuda.empty_cache()

    autoencoder = load_same_l(device, args.same_config, args.same_checkpoint)
    audio = decode(autoencoder, latents, args.batch_size)
    write_wavs(audio, args.out)
    # Keep non-audio files out of the clip directory: the FAD scorer reads every file there.
    (args.out / "meta").mkdir(exist_ok=True)
    (args.out / "meta" / "prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    print(f"wrote {len(prompts)} clips to {args.out}")


if __name__ == "__main__":
    main()
