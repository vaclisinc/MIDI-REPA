"""Clean-endpoint extrapolation along the sampling trajectory (paper Fig. 2, Sec. 3.4).

Sampling runs normally from t = 1 to a chosen step; there the current guided velocity is
frozen and extrapolated to the clean endpoint, ``z0_hat(t) = z_t - t * v_cfg(z_t, t, c)``,
and decoded. This shows the musical content implied at t rather than the residual noise.
The paper's figure uses the baseline EMA checkpoint, seed 42, 25 Euler steps, CFG 4.5,
the second ATTM test prompt, and the steps t = 1.00 ... 0.76, 0.04, 0.

    python -m midi_repa.eval.timestep --checkpoint runs/baseline/ema_final.pth \
        --prompts prompts.txt --out timestep/ [--figure-prompt 1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from midi_repa.autoencoder import decode, load_same_l
from midi_repa.generate import SAMPLE_RATE, load_generator
from midi_repa.text import TextEncoder

FIGURE_STEPS = [0, 1, 2, 3, 4, 5, 6, 25]  # t = 1.00, 0.96, ..., 0.76, then 0.00
N_FFT, HOP, N_MELS, FMIN, FMAX = 2048, 512, 128, 30.0, 16_000.0


@torch.no_grad()
def clean_estimates(net, conditions, empty, x0, cfg_strength: float, num_steps: int,
                    wanted: set[int]) -> dict[int, tuple[float, torch.Tensor]]:
    """``{step: (t, z0_hat)}``; step ``num_steps`` is the finished sample (t = 0)."""
    x = x0
    steps = torch.linspace(1.0, 0.0, num_steps + 1)
    captured = {}
    for index, t in enumerate(steps[:-1]):
        flow = net.ode_wrapper(t, x, conditions, empty, cfg_strength)
        if index in wanted:
            captured[index] = (float(t), (x - float(t) * flow).clone())
        x = x + (steps[index + 1] - t) * flow
    if num_steps in wanted:
        captured[num_steps] = (0.0, x.clone())
    return captured


def log_mel(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    """Hann STFT (2048 / 512), 128 Slaney-normalised mel bands 30 Hz-16 kHz, natural log."""
    wave = np.ascontiguousarray(wave.mean(axis=1) if wave.ndim == 2 else wave, dtype=np.float64)
    window = np.hanning(N_FFT + 1)[:-1]
    frames = 1 + (len(wave) - N_FFT) // HOP
    strided = np.lib.stride_tricks.as_strided(
        wave, shape=(frames, N_FFT), strides=(wave.strides[0] * HOP, wave.strides[0]))
    magnitude = np.abs(np.fft.rfft(strided * window, axis=1)).T
    to_mel = lambda f: 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)
    to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    bins = np.fft.rfftfreq(N_FFT, 1.0 / sample_rate)
    edges = to_hz(np.linspace(to_mel(FMIN), to_mel(min(FMAX, sample_rate / 2)), N_MELS + 2))
    bank = np.zeros((N_MELS, len(bins)))
    for i in range(N_MELS):
        low, centre, high = edges[i], edges[i + 1], edges[i + 2]
        bank[i] = np.clip(np.minimum((bins - low) / max(centre - low, 1e-9),
                                     (high - bins) / max(high - centre, 1e-9)), 0.0, None)
    bank *= (2.0 / np.maximum(edges[2:] - edges[:-2], 1e-9))[:, None]
    return np.log(bank @ magnitude + 1e-8)


def plot(mels: dict[int, np.ndarray], times: dict[int, float], steps: list[int], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
    # One colour scale for every panel; no per-panel normalisation.
    limits = (min(float(np.percentile(m, 5)) for m in mels.values()),
              max(float(m.max()) for m in mels.values()))
    columns = 4
    rows = int(np.ceil(len(steps) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.45, 0.68 * rows + 0.5), squeeze=False)
    for index, (ax, step) in enumerate(zip(axes.flat, steps)):
        ax.imshow(mels[step], origin="lower", aspect="auto", cmap="magma", vmin=limits[0],
                  vmax=limits[1], rasterized=True, extent=[0, 10, 0, N_MELS])
        ax.set_title(f"$t={times[step]:.2f}$", fontsize=9.2, pad=2)
        ax.set_yticks([])
        if index // columns == rows - 1:
            ax.set_xticks([0, 5, 10])
        else:
            ax.set_xticks([])
    for ax in list(axes.flat)[len(steps):]:
        ax.axis("off")
    if len(steps) > 1 and steps[-1] - steps[-2] > 1:  # skipped steps before the last panel
        last = list(axes.flat)[len(steps) - 1]
        last.annotate("...", xy=(-0.15, 0.55), xycoords="axes fraction", ha="center",
                      va="center", fontsize=9)
    fig.supxlabel("Time (s)", fontsize=8, y=0.0)
    fig.supylabel("Mel frequency", fontsize=8, x=0.0)
    fig.subplots_adjust(left=.06, right=.98, bottom=.2, top=.9, wspace=.3, hspace=.55)
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=8, help="first n prompts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-strength", type=float, default=4.5)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--steps", type=int, nargs="+", default=list(range(26)))
    parser.add_argument("--figure-prompt", type=int, default=1)
    parser.add_argument("--same-config", type=Path, default=None)
    parser.add_argument("--same-checkpoint", type=Path, default=None)
    parser.add_argument("--clap-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    device = "cuda"
    prompts = [l for l in args.prompts.read_text(encoding="utf-8").splitlines() if l][:args.n]
    text_f, text_f_c = TextEncoder(device, args.clap_checkpoint)(prompts)
    net = load_generator(args.checkpoint, device)
    conditions = net.preprocess_conditions(text_f, text_f_c)
    empty = net.get_empty_conditions(len(prompts))
    # The same prior draw as midi_repa.generate, so step 25 is the published clip.
    x0 = torch.empty(len(prompts), net.latent_seq_len, net.latent_dim, device=device).normal_(
        generator=torch.Generator(device=device).manual_seed(args.seed))
    captured = clean_estimates(net, conditions, empty, x0, args.cfg_strength, args.num_steps,
                               set(args.steps))
    autoencoder = load_same_l(device, args.same_config, args.same_checkpoint)

    manifest, mels, times = [], {}, {}
    for step in sorted(captured):
        t, latent = captured[step]
        audio = decode(autoencoder, net.unnormalize(latent.clone()))
        for i, wave in enumerate(audio):
            path = args.out / f"prompt_{i:02d}" / f"step{step:02d}_t{t:.2f}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            wave = wave.transpose(0, 1).numpy()
            sf.write(str(path), wave, SAMPLE_RATE)
            manifest.append({"prompt_index": i, "step": step, "t": t, "path": str(path)})
            if i == args.figure_prompt:
                mels[step], times[step] = log_mel(wave, SAMPLE_RATE), t
    (args.out / "manifest.json").write_text(json.dumps(
        {"prompts": prompts, "seed": args.seed, "clips": manifest}, indent=2) + "\n")
    shown = [s for s in FIGURE_STEPS if s in mels]
    if shown:
        plot(mels, times, shown, args.out / "timestep.pdf")
        print(f"figure -> {args.out / 'timestep.pdf'}")


if __name__ == "__main__":
    main()
