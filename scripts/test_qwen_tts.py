from __future__ import annotations

import argparse
from pathlib import Path
import time

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "data" / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
OUTPUT_PATH = ROOT / "data" / "output" / "qwen-sohee-expressive.wav"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--instruct", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾지 못했습니다.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        str(MODEL_PATH),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    loaded_at = time.perf_counter()
    wavs, sample_rate = model.generate_custom_voice(
        text=args.text,
        language="Korean",
        speaker="Sohee",
        instruct=args.instruct,
    )
    sf.write(args.output, wavs[0], sample_rate)
    completed_at = time.perf_counter()
    peak_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"output={args.output}")
    print(f"sample_rate={sample_rate}")
    print(f"load_seconds={loaded_at - started_at:.2f}")
    print(f"generation_seconds={completed_at - loaded_at:.2f}")
    print(f"peak_vram_mib={peak_mib:.0f}")


if __name__ == "__main__":
    main()
