<div align="center">

# MIDI-REPA

### Improving Text-to-Music Generation with Symbolic Representation Alignment

Song-Ze Yu, Fang-Duo Tsai, Yi-Hsuan Yang

<sub>National Taiwan University</sub>

<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2610.20967-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.20967)
[![AMTFlow](https://img.shields.io/badge/AMTFlow-GitHub-181717.svg?logo=github&logoColor=white)](https://github.com/vaclisinc/AMTFlow)
[![JamendoMIDI](https://img.shields.io/badge/JamendoMIDI-Hugging_Face-ffcc4d.svg?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/vaclis/JamendoMIDI)

</div>

<p align="center">
  <a href="https://midi-repa.vaclis.net/">
    <img src="assets/paper-figure.webp" alt="MIDI-REPA overview: an AMT system turns training audio into symbolic targets that are aligned with an early FluxAudio-S DiT layer, with panels on what, where, and when to align" width="100%">
  </a>
  <br/>
  <sub>Click the figure to open the demo page with audio examples.</sub>
</p>

## Code & Resources

<table>
  <tr>
    <td width="50%" valign="top">
      <a href="https://github.com/vaclisinc/AMTFlow"><img src="assets/amtflow-pipeline.webp" alt="AMTFlow pipeline diagram" width="100%"></a>
      <br/><sub>PIPELINE</sub>
      <br/><a href="https://github.com/vaclisinc/AMTFlow"><b>AMTFlow ↗</b></a>
      <br/>Transcription pipeline used for the paper. To reproduce, run with <code>--config paper</code>.
    </td>
    <td width="50%" valign="top">
      <a href="https://huggingface.co/datasets/vaclis/JamendoMIDI"><img src="assets/jamendomidi-viewer.webp" alt="JamendoMIDI dataset viewer" width="100%"></a>
      <br/><sub>DATASET</sub>
      <br/><a href="https://huggingface.co/datasets/vaclis/JamendoMIDI"><b>JamendoMIDI ↗</b></a>
      <br/>Multi-instrument MIDI transcriptions of 51,077 instrumental Jamendo tracks. CC BY-NC-SA 4.0.
    </td>
  </tr>
</table>

## Setup

```bash
uv sync --all-extras
uv run bash scripts/setup_third_party.sh
uv venv --python 3.11 .venv-eval   # official FAD/CLAP scripts
uv pip install --python .venv-eval -r envs/eval-requirements.txt --index-strategy unsafe-best-match
```

SAME-L comes from [`stabilityai/stable-audio-3-medium`](https://huggingface.co/stabilityai/stable-audio-3-medium) (gated; log in first).

## Data

You need [JamendoMIDI](https://huggingface.co/datasets/vaclis/JamendoMIDI), the vocal-removed
Jamendo audio from the [ATTM preprocessing](https://github.com/ntu-musicailab/ICME26-ATTM-GC-Preprocessing),
the ATTM training captions, and the [Song Describer](https://zenodo.org/records/10072001) captions.

```bash
DATA=/path/to/prepared AUDIO=/path/to/vocal_removed_mp3
uv run python -m midi_repa.prep jamendomidi --root $DATA
uv run python -m midi_repa.prep latents     --root $DATA --audio-dir $AUDIO
uv run python -m midi_repa.prep manifest    --root $DATA --qwen-captions qwen.json --flamingo-captions music_flamingo.json
uv run python -m midi_repa.prep rolls       --root $DATA
uv run python -m midi_repa.prep text        --root $DATA
uv run python -m midi_repa.prep basic-pitch --root $DATA --audio-dir $AUDIO   # bp_roll_l4 only
```

## Training

```bash
uv run torchrun --nproc_per_node=1 -m midi_repa.train configs/arms/mus_roll_l2.yaml \
    data.root=$DATA output_dir=runs/mus_roll_l2
```

| Paper | Arms in `configs/arms/` |
|---|---|
| Table 1 | `cube_l4`, `mus_roll_l4`, `bp_roll_l4`, `baseline` |
| Table 3 | `mus_roll_l{0,2,4,5,8,9,11}` |
| Table 4 | `mus_roll_l2` |
| Table 5 | `mus_roll_l2`, `mus_roll_l2_t{088_100,084_100,080_100,000_080,000_020}` |

## Evaluation

SDD (Tables 1, 3, 5), for seeds 42 1234 7777 9999 31415 6798 2024 8888:

```bash
uv run python -m midi_repa.eval.sdd reference --audio-dir $AUDIO --out sdd_ref
uv run python -m midi_repa.generate --checkpoint runs/$ARM/ema_final.pth \
    --prompts assets/eval/sdd_prompts.txt --seed $SEED --out out/$ARM/$SEED
.venv-eval/bin/python -m midi_repa.eval.official fad sdd_ref out/$ARM/$SEED -o fad.csv
.venv-eval/bin/python -m midi_repa.eval.official clap assets/eval/sdd_prompts.csv out/$ARM/$SEED --audio_files_extension .wav -o clap.csv
uv run python -m midi_repa.eval.summarize scores.csv   # arm,seed,fad,clap -> mean and 95% CI
```

ATTM (Table 4):

```bash
uv run python -m midi_repa.eval.attm reference --audio-dir $AUDIO --out attm_ref
uv run python -m midi_repa.generate --checkpoint runs/mus_roll_l2/ema_final.pth \
    --prompts assets/eval/attm_prompts.txt --seed 6798 --out out/attm
uv run python -m midi_repa.eval.attm subset --clips out/attm --out out/attm_n80
.venv-eval/bin/python -m midi_repa.eval.official fad attm_ref out/attm_n80 -o attm_fad.csv
.venv-eval/bin/python -m midi_repa.eval.official clap out/attm_n80_prompts.csv out/attm_n80 --audio_files_extension .wav -o attm_clap.csv
uv run python -m midi_repa.eval.ccs --clips out/attm --out ccs.json
```

FoG-A (Table 2) and Fig. 2:

```bash
uv run python -m midi_repa.eval.foga --checkpoint runs/baseline/ckpt_300000.pth data.root=$DATA --out foga.json
uv run python -m midi_repa.eval.timestep --checkpoint runs/baseline/ema_final.pth \
    --prompts assets/eval/attm_prompts.txt --out timestep/
```

GPU generation is not bit-exact across machines; scores match the paper to about 0.0005.

## License

MIT. JamendoMIDI is CC BY-NC-SA 4.0.

## Citation

```bibtex
@misc{yu2026midirepa,
      title={MIDI-REPA: Improving Text-to-Music Generation with Symbolic Representation Alignment},
      author={Song-Ze Yu and Fang-Duo Tsai and Yi-Hsuan Yang},
      year={2026},
      eprint={2610.20967},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2610.20967},
}
```
