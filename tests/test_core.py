"""CPU tests of the pieces whose exact behaviour the paper's numbers depend on."""

from pathlib import Path

import pretty_midi
import pytest
import torch

from midi_repa.config import load_config
from midi_repa.data import PITCHES, TEACHERS
from midi_repa.model import build_model
from midi_repa.prep.manifest import caption_source, validation_window
from midi_repa.prep.rolls import merge_cube, midi_to_cube
from midi_repa.repa import BlockAblate, BlockTap, RepaHead, repa_loss

ARMS = sorted((Path(__file__).resolve().parents[1] / "configs" / "arms").glob("*.yaml"))


def test_fifteen_arms_load():
    assert len(ARMS) == 15
    for arm in ARMS:
        cfg = load_config(arm, ["data.root=/data"])
        assert cfg.run_name == arm.stem
        if cfg.repa.enabled:
            assert cfg.repa.teacher in TEACHERS and 0 <= cfg.repa.layer < 12


def test_fluxaudio_s_size_and_shapes():
    net = build_model("fluxaudio_s_same", use_rope=True, text_c_dim=512).eval()
    assert sum(p.numel() for p in net.parameters()) == 121_666_688
    out = net(torch.randn(2, 108, 256), torch.randn(2, 77, 1024), torch.randn(2, 512),
              torch.rand(2))
    assert out.shape == (2, 108, 256)


def test_tap_and_ablate_hit_the_right_blocks():
    net = build_model("fluxaudio_s_same", use_rope=True, text_c_dim=512).eval()
    inputs = (torch.randn(1, 108, 256), torch.randn(1, 77, 1024), torch.randn(1, 512),
              torch.rand(1))
    taps = {layer: BlockTap(net, layer) for layer in (0, 2, 4, 11)}
    with torch.no_grad():
        net(*inputs)
    assert all(tap.latest.shape == (1, 108, 448) for tap in taps.values())
    assert taps[0].module is net.joint_blocks[0] and taps[4].module is net.fused_blocks[0]
    # The initialisation zeroes the adaLN gates and the output layer, so give every
    # parameter a random value before checking that ablating one block changes the output.
    torch.manual_seed(0)
    for parameter in net.parameters():
        if parameter.requires_grad:
            torch.nn.init.normal_(parameter, std=0.02)
    ablate = BlockAblate(net, 7)
    with torch.no_grad():
        full = net(*inputs)
        ablate.enabled = True
        without = net(*inputs)
    assert not torch.allclose(full, without)


def test_repa_loss_ignores_silent_frames():
    target = torch.zeros(1, 4, PITCHES)
    target[0, 0, 40] = 1
    target[0, 2, 50] = 1
    pred = torch.randn(1, 4, PITCHES)
    pred[0, 0] = target[0, 0]
    pred[0, 2] = target[0, 2]
    loss, frac = repa_loss(pred, target)
    assert loss.item() == pytest.approx(-1.0, abs=1e-6)  # silent frames 1 and 3 excluded
    assert frac.item() == pytest.approx(0.5)


def test_repa_head_is_one_temporal_conv():
    head = RepaHead(448, 88)
    assert [type(m).__name__ for m in head.net] == ["Conv1d"]
    assert head.net[0].kernel_size == (3,) and head.net[0].padding == (1,)
    assert head(torch.randn(2, 108, 448)).shape == (2, 108, 88)


def test_rolls_from_midi(tmp_path):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name="acoustic piano")
    piano.notes.append(pretty_midi.Note(100, 60, start=0.0, end=0.2))     # frames 0-2
    piano.notes.append(pretty_midi.Note(100, 64, start=10.05, end=10.1))  # window 1, frame 0-1
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    drums.notes.append(pretty_midi.Note(100, 36, start=0.0, end=0.05))
    midi.instruments += [piano, drums]
    path = tmp_path / "x.mid"
    midi.write(str(path))
    cube = midi_to_cube(path, windows=2)
    assert cube.shape == (2, 108, 7 * PITCHES)
    piano_slot, drum_slot = 3, 0  # assets/instrument_palette.json order
    assert cube[0, :3, piano_slot * PITCHES + 60 - 21].all() and not cube[0, 3, piano_slot * PITCHES + 39]
    assert cube[1, 0, piano_slot * PITCHES + 64 - 21] == 1
    assert cube[0, 0, drum_slot * PITCHES + 36 - 21] == 1
    roll = merge_cube(cube)
    assert roll.shape == (2, 108, PITCHES) and roll[0, 0, 36 - 21] == 1 and roll[0, 0, 60 - 21] == 1


def test_split_hashing_matches_paper_manifests():
    # Rows of the paper's manifests.
    assert caption_source("768") == "qwen"
    assert caption_source("2208") == "music_flamingo"
    assert validation_window("768", 29) == 6
    assert validation_window("2208", 21) == 8


def test_attm_assets_agree():
    import csv

    eval_dir = Path(__file__).resolve().parents[1] / "assets" / "eval"
    prompts = (eval_dir / "attm_prompts.txt").read_text(encoding="utf-8").splitlines()
    with (eval_dir / "attm_prompts.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (eval_dir / "attm_ccs_metadata.csv").open(encoding="utf-8", newline="") as handle:
        questions = list(csv.DictReader(handle))
    assert len(prompts) == 100 and [r["caption"] for r in rows] == prompts
    assert [r["id"] for r in rows] == [f"gen_{i}" for i in range(100)]
    assert len(questions) == 300
    assert all(q["caption"] == prompts[int(q["id"])] for q in questions)
    assert len(set((eval_dir / "attm_reference_ids.txt").read_text().split())) == 1000
