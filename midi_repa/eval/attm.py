"""ICME 2026 ATTM benchmark (paper Table 4).

The 100 official test prompts are ``assets/eval/attm_prompts.txt`` (``gen_<i>.wav`` belongs
to line ``i``; ``attm_prompts.csv`` is the same list for the official CLAP script) and the
CCS question metadata is ``attm_ccs_metadata.csv``. All 100 prompts are generated; FAD, CLAP
and CCS are reported on prompts 0-79. The FAD reference is the challenge's 1,000
vocal-removed Jamendo tracks listed in ``attm_reference_ids.txt``.

    python -m midi_repa.eval.attm reference --audio-dir <vocal-removed mp3 dir> --out attm_ref
    python -m midi_repa.eval.attm subset --clips samples/attm --out samples/attm_n80
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "eval"
N_SCORED = 80


def build_reference(audio_dir: Path, out: Path) -> None:
    """One link per reference track to its vocal-removed audio."""
    track_ids = (ASSETS / "attm_reference_ids.txt").read_text().split()
    out.mkdir(parents=True, exist_ok=True)
    for track_id in track_ids:
        source = Path(audio_dir).resolve() / f"{track_id}.mp3"
        if not source.is_file():
            raise FileNotFoundError(source)
        link = out / f"{track_id}.mp3"
        if not link.exists():
            link.symlink_to(source)
    print(f"{len(track_ids)} reference tracks -> {out}")


def build_subset(clips: Path, out: Path, n: int = N_SCORED) -> None:
    """Links to ``gen_0`` ... ``gen_<n-1>`` and the matching CLAP prompt file."""
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        source = Path(clips).resolve() / f"gen_{i}.wav"
        if not source.is_file():
            raise FileNotFoundError(source)
        link = out / source.name
        if not link.exists():
            link.symlink_to(source)
    with (ASSETS / "attm_prompts.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))[:n]
    with (out.parent / f"{out.name}_prompts.csv").open("w", encoding="utf-8",
                                                       newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "caption"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{n} clips -> {out}, prompts -> {out.parent / f'{out.name}_prompts.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    reference = sub.add_parser("reference", help="FAD reference directory")
    reference.add_argument("--audio-dir", type=Path, required=True)
    reference.add_argument("--out", type=Path, required=True)
    subset = sub.add_parser("subset", help="the scored clips 0-79 and their CLAP prompts")
    subset.add_argument("--clips", type=Path, required=True)
    subset.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "reference":
        build_reference(args.audio_dir, args.out)
    else:
        build_subset(args.clips, args.out)


if __name__ == "__main__":
    main()
