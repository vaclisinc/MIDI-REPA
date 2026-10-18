"""Run the official ICME 2026 ATTM ``src/fad.py`` / ``src/clap.py`` unmodified.

Run this with the evaluation environment (see README), not the training one:

    .venv-eval/bin/python -m midi_repa.eval.official fad  <reference_dir> <clip_dir> -w 8 -o fad.csv
    .venv-eval/bin/python -m midi_repa.eval.official clap <id2caption.csv> <clip_dir> -o clap.csv

FAD uses LAION-CLAP music embeddings (fadtk); CLAP uses the music/speech/AudioSet model.
Two runtime guards only, no change to any computed value:

* ``laion_clap.load_ckpt`` logs with three positional arguments and no format string,
  which raises inside fadtk's workers once fadtk installs a formatting handler. INFO
  logging is disabled before the workers start.
* fadtk's statistics step starts a large process pool; it is replaced by an ordered
  thread pool with the same per-file moments and the same ordered reduction.
"""

from __future__ import annotations

import logging
import runpy
import sys
from concurrent.futures import ThreadPoolExecutor

from midi_repa.third_party import path


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("fad", "clap"):
        raise SystemExit(__doc__)
    metric, args = sys.argv[1], sys.argv[2:]
    root = path("attm-eval")
    if metric == "fad":
        workers = 8
        for flag in ("-w", "--workers"):
            if flag in args:
                workers = int(args[args.index(flag) + 1])
        import fadtk.utils

        def ordered_threads(function, values, *unused, **kwargs):
            with ThreadPoolExecutor(max_workers=workers) as executor:
                return list(executor.map(function, values))

        fadtk.utils.pmap = ordered_threads
    logging.disable(logging.INFO)
    script = root / "src" / f"{metric}.py"
    if metric == "clap":
        # clap.py downloads and reads its checkpoint under ./load/, so it runs from the
        # checkout; make path arguments absolute first.
        import os
        from pathlib import Path
        args = [str(Path(a).resolve()) if Path(a).exists() or a.endswith(".csv") else a
                for a in args]
        os.chdir(root)
    sys.argv = [str(script), *args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
