"""Song Describer (SDD) design-study benchmark (paper Sec. 3.1).

The 706 SDD tracks are withheld from training. Captions that ask for a voice are dropped,
since the training corpus is vocal-removed, leaving 629 prompts from 475 tracks
(``assets/eval/sdd_prompts.txt``; ``gen_<i>.wav`` belongs to line ``i``).

The FAD reference is caption-matched: one copy of a track's (vocal-removed) audio per
retained caption, so 629 entries over 475 tracks. ``reference`` materialises it as a
directory of links for the official FAD script.

    python -m midi_repa.eval.sdd prompts   --song-describer-csv song_describer.csv
    python -m midi_repa.eval.sdd reference --audio-dir <vocal-removed mp3 dir> --out sdd_ref
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "eval"

VOCAL_PATTERN = re.compile(
    r"\b("
    r"vocal|vocals|vocalist|vocalists|vocalise|vocalization|"
    r"sing|sings|singing|singer|singers|sung|"
    r"lyric|lyrics|lyrical|"
    r"voice|voices|voiced|"
    r"rap|rapping|rapper|rappers|"
    r"chant|chanting|choir|acapella|a cappella|"
    r"male|female|falsetto|humming|hums"
    r")\b",
    re.IGNORECASE,
)


def build_prompts(song_describer_csv: Path, out_dir: Path) -> list[tuple[str, str, str]]:
    """Filter ``song_describer.csv`` (Zenodo 10072001) to the instrumental captions."""
    track_ids = set((ASSETS / "sdd_track_ids.txt").read_text().split())
    kept = []
    with Path(song_describer_csv).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            track_id = str(row["track_id"]).strip()
            caption = " ".join(str(row["caption"]).split())
            if track_id in track_ids and not VOCAL_PATTERN.search(caption):
                kept.append((str(row["caption_id"]), track_id, caption))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sdd_prompts.txt").write_text("\n".join(c for _, _, c in kept) + "\n",
                                             encoding="utf-8")
    with (out_dir / "sdd_prompts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "caption"])
        writer.writerows([f"gen_{i}", caption] for i, (_, _, caption) in enumerate(kept))
    with (out_dir / "sdd_reference.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prompt_index", "caption_id", "track_id"])
        writer.writerows([i, caption_id, track] for i, (caption_id, track, _) in enumerate(kept))
    print(f"kept {len(kept)} captions from {len({t for _, t, _ in kept})} tracks -> {out_dir}")
    return kept


def build_reference(audio_dir: Path, out: Path, reference_csv: Path | None = None) -> None:
    """One link per retained caption to the vocal-removed audio of its track."""
    reference_csv = reference_csv or ASSETS / "sdd_reference.csv"
    out.mkdir(parents=True, exist_ok=True)
    with Path(reference_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        source = Path(audio_dir).resolve() / f"{row['track_id']}.mp3"
        if not source.is_file():
            raise FileNotFoundError(source)
        link = out / f"ref_{int(row['prompt_index']):04d}_{row['track_id']}.mp3"
        if not link.exists():
            link.symlink_to(source)
    print(f"{len(rows)} reference entries ({len({r['track_id'] for r in rows})} tracks) -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    prompts = sub.add_parser("prompts", help="rebuild assets/eval/sdd_prompts.* from the SDD csv")
    prompts.add_argument("--song-describer-csv", type=Path, required=True)
    prompts.add_argument("--out", type=Path, default=ASSETS)
    reference = sub.add_parser("reference", help="caption-matched FAD reference directory")
    reference.add_argument("--audio-dir", type=Path, required=True)
    reference.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prompts":
        build_prompts(args.song_describer_csv, args.out)
    else:
        build_reference(args.audio_dir, args.out)


if __name__ == "__main__":
    main()
