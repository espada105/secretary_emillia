from __future__ import annotations

import subprocess
import threading
import uuid
import wave
from datetime import datetime
import re
from pathlib import Path

from .config import Settings
from .enums import TTSEngine, VoiceName
from .qwen_worker import PersistentQwenWorker


RVC_MODELS: dict[VoiceName, tuple[str, str]] = {
    VoiceName.EMILIA: ("Emilia.pth", "Emilia_v2.index"),
    VoiceName.RAM: ("Ram.pth", "Ram_v2.index"),
}

QWEN_MODELS: dict[TTSEngine, tuple[str, bool]] = {
    TTSEngine.QWEN3: ("Qwen3-TTS-12Hz-1.7B-CustomVoice", True),
    TTSEngine.QWEN3_KOREAN: ("Qwen3-TTS-12Hz-0.6B-Korean", False),
}

NATIVE_QWEN_SPEAKERS: dict[TTSEngine, dict[VoiceName, str]] = {
    TTSEngine.QWEN3: {
        VoiceName.BASE_KOREAN: "Sohee",
        VoiceName.ONO_ANNA: "Ono_Anna",
    },
    TTSEngine.QWEN3_KOREAN: {
        VoiceName.BASE_KOREAN: "korean_tts",
    },
}


class ExpressiveTTS:
    """Runs the Qwen TTS and optional RVC stages in WSL, one at a time."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._project_root = Path(__file__).resolve().parents[2]
        self._lock = threading.Lock()
        self._qwen_worker = PersistentQwenWorker(settings, self._project_root)

    @staticmethod
    def _to_wsl_path(path: Path) -> str:
        resolved = path.resolve()
        drive = resolved.drive.rstrip(":").lower()
        if len(drive) != 1:
            raise RuntimeError(f"WSL 경로로 바꿀 수 없습니다: {resolved}")
        suffix = resolved.as_posix().split(":", maxsplit=1)[1]
        return f"/mnt/{drive}{suffix}"

    def _run_wsl(self, script_name: str, *arguments: str) -> None:
        script = self._to_wsl_path(self._project_root / "scripts" / script_name)
        command = [
            "wsl.exe",
            "-d",
            self._settings.secretary_wsl_distro,
            "--",
            "bash",
            script,
            *arguments,
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=360,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("WSL을 찾을 수 없습니다. Qwen/RVC는 WSL 환경이 필요합니다.") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("음성 생성 시간이 초과되었습니다.") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "알 수 없는 WSL 오류").strip()
            raise RuntimeError(detail[-1200:]) from exc
        if completed.stderr:
            # Qwen/Torch warnings are normal, but preserve actionable errors.
            print(completed.stderr)

    @staticmethod
    def _pad_short_rvc_input(source: Path, minimum_seconds: float = 4.0) -> Path:
        """RVC needs a few seconds of context; preserve terse assistant replies."""
        with wave.open(str(source), "rb") as audio:
            parameters = audio.getparams()
            frames = audio.readframes(audio.getnframes())
        duration = parameters.nframes / parameters.framerate
        if duration >= minimum_seconds:
            return source

        padded = source.with_name(f"{source.stem}.padded.wav")
        remaining_frames = int((minimum_seconds - duration) * parameters.framerate)
        silence = b"\x00" * remaining_frames * parameters.nchannels * parameters.sampwidth
        with wave.open(str(padded), "wb") as audio:
            audio.setparams(parameters)
            audio.writeframes(frames + silence)
        return padded

    def synthesize(
        self,
        *,
        text: str,
        engine: TTSEngine,
        voice: VoiceName,
        tone: str,
        instruct: str,
        index_rate: float,
        protect: float,
        pitch: int,
    ) -> tuple[str, Path, float]:
        normalized = " ".join(text.split())
        if not normalized:
            raise ValueError("텍스트를 입력해 주세요.")
        if len(normalized) > 500:
            raise ValueError("테스트 입력은 500자 이하로 제한됩니다.")
        normalized_tone = re.sub(r"[^a-z0-9_-]", "", tone.lower())[:24] or "custom"
        if engine not in QWEN_MODELS:
            raise ValueError(f"지원하지 않는 Qwen 엔진입니다: {engine}")
        native_speakers = NATIVE_QWEN_SPEAKERS[engine]
        if voice not in {*native_speakers, *RVC_MODELS}:
            raise ValueError(f"아직 준비되지 않은 음성입니다: {voice}")
        if voice == VoiceName.ONO_ANNA and engine != TTSEngine.QWEN3:
            raise ValueError("Ono_Anna는 Qwen3 1.7B CustomVoice 엔진에서만 사용할 수 있습니다.")

        output_dir = self._settings.secretary_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        audio_id = (
            f"{engine}-{voice}-tone{normalized_tone}-pitch{pitch:+d}-strength{index_rate:.2f}-protect{protect:.2f}"
            f"-{stamp}-{uuid.uuid4().hex[:8]}"
        )
        source = output_dir / f".{audio_id}.qwen.wav"
        destination = output_dir / f"{audio_id}.wav"
        model_directory, supports_instruct = QWEN_MODELS[engine]
        speaker = native_speakers.get(voice, native_speakers[VoiceName.BASE_KOREAN])

        with self._lock:
            self._qwen_worker.generate(
                text=normalized,
                instruct=instruct if supports_instruct else "",
                output=source,
                model=self._project_root / "data" / "models" / model_directory,
                speaker=speaker,
            )
            if voice in native_speakers:
                source.replace(destination)
            else:
                model_name, index_name = RVC_MODELS[voice]
                rvc_input = self._pad_short_rvc_input(source)
                try:
                    self._run_wsl(
                        "run_rvc_wsl.sh",
                        "--model",
                        self._to_wsl_path(self._project_root / "models" / "rezero-lim" / "extracted" / model_name),
                        "--input",
                        self._to_wsl_path(rvc_input),
                        "--output",
                        self._to_wsl_path(destination),
                        "--index",
                        self._to_wsl_path(
                            self._project_root / "models" / "rezero-lim" / "extracted" / "Index" / index_name
                        ),
                        "--index-rate",
                        str(index_rate),
                        "--protect",
                        str(protect),
                        "--pitch",
                        str(pitch),
                    )
                finally:
                    if rvc_input != source:
                        rvc_input.unlink(missing_ok=True)
                    source.unlink(missing_ok=True)

        try:
            with wave.open(str(destination), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
                if audio.getnchannels() != 1 or duration < 0.2:
                    raise RuntimeError("유효하지 않은 WAV가 생성되었습니다.")
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return audio_id, destination, duration

    def shutdown(self) -> None:
        self._qwen_worker.shutdown()

    def warmup_default_model(self) -> None:
        model_directory, _supports_instruct = QWEN_MODELS[TTSEngine.QWEN3]
        self._qwen_worker.warmup(model=self._project_root / "data" / "models" / model_directory)
