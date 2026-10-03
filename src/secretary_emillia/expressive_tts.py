from __future__ import annotations

import subprocess
import threading
import uuid
import wave
from datetime import datetime
from pathlib import Path

from .config import Settings
from .enums import TTSEngine, VoiceName


RVC_MODELS: dict[VoiceName, tuple[str, str]] = {
    VoiceName.EMILIA: ("Emilia.pth", "Emilia_v2.index"),
    VoiceName.RAM: ("Ram.pth", "Ram_v2.index"),
}

QWEN_MODELS: dict[TTSEngine, tuple[str, str, bool]] = {
    TTSEngine.QWEN3: ("Qwen3-TTS-12Hz-1.7B-CustomVoice", "Sohee", True),
    TTSEngine.QWEN3_KOREAN: ("Qwen3-TTS-12Hz-0.6B-Korean", "korean_tts", False),
}


class ExpressiveTTS:
    """Runs the Qwen TTS and optional RVC stages in WSL, one at a time."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._project_root = Path(__file__).resolve().parents[2]
        self._lock = threading.Lock()

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

    def synthesize(
        self,
        *,
        text: str,
        engine: TTSEngine,
        voice: VoiceName,
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
        if engine not in QWEN_MODELS:
            raise ValueError(f"지원하지 않는 Qwen 엔진입니다: {engine}")
        if voice not in {VoiceName.BASE_KOREAN, *RVC_MODELS}:
            raise ValueError(f"아직 준비되지 않은 음성입니다: {voice}")

        output_dir = self._settings.secretary_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        audio_id = (
            f"{engine}-{voice}-pitch{pitch:+d}-strength{index_rate:.2f}-protect{protect:.2f}"
            f"-{stamp}-{uuid.uuid4().hex[:8]}"
        )
        source = output_dir / f".{audio_id}.qwen.wav"
        destination = output_dir / f"{audio_id}.wav"
        model_directory, speaker, supports_instruct = QWEN_MODELS[engine]

        with self._lock:
            self._run_wsl(
                "run_qwen_tts_wsl.sh",
                "--text",
                normalized,
                "--instruct",
                instruct if supports_instruct else "",
                "--output",
                self._to_wsl_path(source),
                "--model",
                self._to_wsl_path(self._project_root / "data" / "models" / model_directory),
                "--speaker",
                speaker,
            )
            if voice == VoiceName.BASE_KOREAN:
                source.replace(destination)
            else:
                model_name, index_name = RVC_MODELS[voice]
                self._run_wsl(
                    "run_rvc_wsl.sh",
                    "--model",
                    self._to_wsl_path(self._project_root / "models" / "rezero-lim" / "extracted" / model_name),
                    "--input",
                    self._to_wsl_path(source),
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
