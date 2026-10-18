"""Data preparation, in order (see README "Data"):

    python -m midi_repa.prep jamendomidi --root DATA
    python -m midi_repa.prep latents     --root DATA --audio-dir AUDIO [--worker i --workers n]
    python -m midi_repa.prep manifest    --root DATA --qwen-captions qwen.json \
                                         --flamingo-captions music_flamingo.json
    python -m midi_repa.prep rolls       --root DATA
    python -m midi_repa.prep text        --root DATA
    python -m midi_repa.prep basic-pitch --root DATA --audio-dir AUDIO   # Table 1 only
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    def stage(name: str, help_text: str):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--root", type=Path, required=True, help="prepared-data root (data.root)")
        return p

    stage("jamendomidi", "download JamendoMIDI MIDI files and splits")
    p = stage("latents", "SAME-L latents of the vocal-removed audio")
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--same-config", type=Path, default=None)
    p.add_argument("--same-checkpoint", type=Path, default=None)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p = stage("manifest", "train/validation manifests")
    p.add_argument("--qwen-captions", type=Path, required=True)
    p.add_argument("--flamingo-captions", type=Path, required=True)
    p = stage("rolls", "MuScriptor cube and merged roll from the MIDI")
    p.add_argument("--workers", type=int, default=16)
    p = stage("text", "T5 + CLAP caption features")
    p.add_argument("--clap-checkpoint", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p = stage("basic-pitch", "Basic Pitch roll from the audio")
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.stage == "jamendomidi":
        from midi_repa.prep.jamendomidi import download
        download(args.root)
    elif args.stage == "latents":
        from midi_repa.prep.latents import build
        build(args.root, args.audio_dir, args.same_config, args.same_checkpoint, args.worker,
              args.workers, args.batch_size)
    elif args.stage == "manifest":
        from midi_repa.prep.manifest import build
        build(args.root, args.qwen_captions, args.flamingo_captions)
    elif args.stage == "rolls":
        from midi_repa.prep.rolls import build
        build(args.root, args.workers)
    elif args.stage == "text":
        from midi_repa.prep.text import build
        build(args.root, args.clap_checkpoint, args.batch_size)
    else:
        from midi_repa.prep.basic_pitch import build
        build(args.root, args.audio_dir, worker=args.worker, workers=args.workers)


if __name__ == "__main__":
    main()
