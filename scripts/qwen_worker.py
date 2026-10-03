from __future__ import annotations

import contextlib
import gc
import json
import sys
import traceback
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


_model: Qwen3TTSModel | None = None
_model_path: str | None = None


def emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def get_model(model_path: str) -> Qwen3TTSModel:
    global _model, _model_path
    if _model is not None and _model_path == model_path:
        return _model
    _model = None
    _model_path = None
    gc.collect()
    torch.cuda.empty_cache()
    with contextlib.redirect_stdout(sys.stderr):
        _model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    _model_path = model_path
    return _model


def generate(request: dict[str, object]) -> None:
    request_id = str(request["id"])
    try:
        text = str(request["text"])
        instruct = str(request.get("instruct", ""))
        output = Path(str(request["output"]))
        speaker = str(request["speaker"])
        with contextlib.redirect_stdout(sys.stderr):
            model = get_model(str(request["model"]))
            arguments = {"text": text, "language": "Korean", "speaker": speaker}
            if instruct:
                arguments["instruct"] = instruct
            wavs, sample_rate = model.generate_custom_voice(**arguments)
            output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output, wavs[0], sample_rate)
        emit({"id": request_id, "ok": True})
    except Exception:
        emit({"id": request_id, "ok": False, "error": traceback.format_exc(limit=3)[-1600:]})


for line in sys.stdin:
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        continue
    if request.get("command") == "shutdown":
        break
    if request.get("command") == "warmup":
        request_id = str(request.get("id", ""))
        try:
            with contextlib.redirect_stdout(sys.stderr):
                get_model(str(request["model"]))
            emit({"id": request_id, "ok": True})
        except Exception:
            emit({"id": request_id, "ok": False, "error": traceback.format_exc(limit=3)[-1600:]})
        continue
    generate(request)
