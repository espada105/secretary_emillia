from __future__ import annotations

import argparse
from pathlib import Path
import time

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "data" / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--instruct", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--speaker", default="Sohee")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾지 못했습니다.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        str(args.model),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    loaded_at = time.perf_counter()
    generate_args = {"text": args.text, "language": "Korean", "speaker": args.speaker}
    if args.instruct:
        generate_args["instruct"] = args.instruct
    wavs, sample_rate = model.generate_custom_voice(**generate_args)
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
