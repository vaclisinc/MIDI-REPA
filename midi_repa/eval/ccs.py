"""Concept Coverage Score (CCS) of the ATTM benchmark with Qwen3-Omni (paper Table 4).

The ATTM metadata (``assets/eval/attm_ccs_metadata.csv``) expands each of the 100 test prompts into one genre, one instrument and
one mood/theme yes/no question. For each question, the answer is "Yes" when the ALM's
log-probability of the token ``Yes`` exceeds that of ``No``. CCS is the fraction of Yes:
240 questions over prompts 0-79 (``ccs_n80``, reported in the paper) and 300 over all 100.

    python -m midi_repa.eval.ccs --clips samples/attm --out ccs.json

Clips are resampled 44.1 -> 16 kHz exactly as in the benchmark's conversion notebook and
only the first channel is given to the model. Needs the ``ccs`` extra and a GPU setup that
holds Qwen3-Omni-30B-A3B in bf16 (about 70 GB; ``--max-gpu-memory`` caps placement).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

METADATA = Path(__file__).resolve().parents[2] / "assets" / "eval" / "attm_ccs_metadata.csv"
MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"


def question(category: str, tag: str) -> str:
    """The benchmark's prompt templates, verbatim."""
    if category == "genre":
        return f'''Please act as a high-sensitivity music genre classifier. Your task is to determine if the genre {tag} is present somewhere in the provided audio.
Answer with exactly one word: "Yes" or "No". Do not provide any explanations or other text.
Scan the audio thoroughly. If you detect any elements like style, rhythmic patterns, or signature instruments, etc. associated with {tag}, answer "Yes". Even if {tag} is only a minor influence or part of a fusion, it counts. Only answer "No" if the audio has absolutely no connection to {tag}.
'''
    if category == "instrument":
        return (
            "\nPlease act as a high-sensitivity audio event detector. Your task is to "
            f"determine if the sound of {tag} is present in the provided audio. \n"
            'Answer with exactly one word: "Yes" or "No". Do not provide any '
            "explanations or other text. \n"
            f"Scan the audio thoroughly from start to finish. If you detect any trace of {tag}, "
            'answer "Yes". Only answer "No" if the sound is 100% absent throughout the entire '
            "duration.\n"
        )
    if category == "mood/theme":
        return f'''Please act as a high-sensitivity music mood/theme classifier. Your task is to determine if the mood/theme {tag} is present in the provided audio.
Answer with exactly one word: "Yes" or "No". Do not provide any explanations or other text.
Scan the audio thoroughly. If there is even a subtle undertone, emotional hint, or atmospheric vibe of {tag}, answer "Yes". I am looking for any trace of this sentiment, regardless of how faint it is. Only answer "No" if the {tag} is 100% irrelevant to the emotional core of the track.
'''
    raise ValueError(f"unknown ATTM category {category!r}")


def load_clip_16k(path: Path) -> torch.Tensor:
    import torchaudio
    import torchaudio.transforms as T

    audio, rate = torchaudio.load(str(path))
    if rate != 44_100:
        raise ValueError(f"{path} is {rate} Hz; the benchmark conversion assumes 44100")
    return T.Resample(orig_freq=44_100, new_freq=16_000)(audio)


@torch.inference_mode()
def yes_no_logprobs(model, processor, audio: torch.Tensor, text: str) -> dict[str, float]:
    """Log-probabilities of the single tokens "Yes" and "No" as the assistant's first token.

    One forward on "<prompt>Yes": under causal attention the distribution at the position
    before the answer depends on the prompt only, so both options are read from it.
    """
    from qwen_omni_utils import process_mm_info

    conversation = [[{"role": "user", "content": [
        {"type": "audio", "audio": audio[0].numpy()}, {"type": "text", "text": text}]}]]
    prompt = processor.apply_chat_template(conversation, tokenize=False,
                                           add_generation_prompt=True)
    audios, _, _ = process_mm_info(conversation, use_audio_in_video=False)
    options = {o: processor.tokenizer(o, add_special_tokens=False).input_ids for o in ("Yes", "No")}
    if any(len(ids) != 1 for ids in options.values()):
        raise RuntimeError(f"expected single-token options, got {options}")
    encoded = processor(text=[p + "Yes" for p in prompt], audio=audios, return_tensors="pt",
                        padding=True, use_audio_in_video=False)
    encoded = encoded.to(model.device).to(model.dtype)
    logits = model.thinker(**encoded).logits

    prefix = processor.tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
    ids = encoded.input_ids[0].tolist()
    start = next(p + len(prefix) for p in range(len(ids) - len(prefix), -1, -1)
                 if ids[p:p + len(prefix)] == prefix)
    logprob = F.log_softmax(logits[0, start - 1, :].float().cpu(), dim=-1)
    return {o: logprob[i[0]].item() for o, i in options.items()}


def main() -> None:
    import pandas as pd
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=Path, required=True, help="gen_<id>.wav for ids 0-99")
    parser.add_argument("--metadata", type=Path, default=METADATA,
                        help="ATTM expanded prompt metadata: id, caption, category, tag")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-gpu-memory", default=None, help="e.g. 70GiB, or 40GiB,40GiB")
    parser.add_argument("--attention", default="sdpa", choices=("sdpa", "flash_attention_2"))
    parser.add_argument("--max-items", type=int, default=None,
                        help="score only the first questions (hardware preflight)")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    if list(metadata.columns) != ["id", "caption", "category", "tag"] or len(metadata) != 300:
        raise SystemExit("expected the 300-row ATTM metadata (id, caption, category, tag)")
    max_memory = None
    if args.max_gpu_memory:
        caps = args.max_gpu_memory.split(",")
        caps = caps * torch.cuda.device_count() if len(caps) == 1 else caps
        max_memory = {**dict(enumerate(caps)), "cpu": "120GiB"}
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype="auto", device_map="auto",
        max_memory=max_memory, attn_implementation=args.attention).eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    rows = metadata.iloc[:args.max_items] if args.max_items else metadata
    results, cache = [], {}
    for row in rows.itertuples():
        clip = int(row.id)
        if clip not in cache:
            cache = {clip: load_clip_16k(args.clips / f"gen_{clip}.wav")}
        scores = yes_no_logprobs(model, processor, cache[clip], question(row.category, row.tag))
        results.append({"id": clip, "category": row.category, "tag": row.tag,
                        "yes_score": scores["Yes"], "no_score": scores["No"]})
    yes = [r["yes_score"] > r["no_score"] for r in results]
    complete = len(results) == 300
    summary = {"ccs_n80": sum(y for y, r in zip(yes, results) if r["id"] < 80) / 240
               if complete else None,
               "ccs_n100": sum(yes) / 300 if complete else None,
               "questions": len(results), "model": MODEL_ID, "revision": MODEL_REVISION,
               "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if complete:
        print(f"CCS N=80 {summary['ccs_n80']:.4f}  N=100 {summary['ccs_n100']:.4f} -> {args.out}")
    else:
        print(f"scored {len(results)} questions (preflight) -> {args.out}")


if __name__ == "__main__":
    main()
