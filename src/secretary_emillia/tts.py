from __future__ import annotations

import threading
import uuid
import wave
import platform
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

from .config import Settings


class KoreanTTS:
    """Lazy, CPU-first adapter around MeloTTS's local Korean model."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            self._apply_windows_compatibility()
            from melo.api import TTS

            self._model = TTS(language="KR", device="cpu")
            self._force_g2pkk_mecab()
        return self._model

    @staticmethod
    def _apply_windows_compatibility() -> None:
        """Avoid DLLs blocked by this Windows workstation's app-control policy.

        MeloTTS imports training/audio helpers at module import time although
        inference does not call them.  Containers use the normal native
        modules; the lightweight fallbacks are only enabled when the host
        blocks llvmlite.
        """
        if platform.system() != "Windows":
            return
        # The Windows mecab-python3 wheel references ``mecab.types`` while
        # shipping files in ``MeCab``.  Supply its expected package alias.
        mecab_alias = types.ModuleType("mecab")
        mecab_alias.__path__ = [str(Path(sys.prefix) / "Lib" / "site-packages" / "MeCab")]
        mecab_alias.__spec__ = ModuleSpec("mecab", loader=None, is_package=True)
        sys.modules.setdefault("mecab", mecab_alias)
        import MeCab as native_mecab

        if not hasattr(native_mecab, "Tagger"):
            native_mecab.Tagger = native_mecab.MeCab
        try:
            import numba  # noqa: F401
        except OSError:
            class _NumbaType:
                def __getitem__(self, _item):
                    return self

            fake_numba = types.ModuleType("numba")
            fake_numba.__spec__ = ModuleSpec("numba", loader=None)
            fake_numba.int32 = _NumbaType()
            fake_numba.float32 = _NumbaType()
            fake_numba.void = lambda *_args: None
            fake_numba.jit = lambda *_args, **_kwargs: lambda function: function
            sys.modules["numba"] = fake_numba

            fake_librosa = types.ModuleType("librosa")
            fake_filters = types.ModuleType("librosa.filters")
            fake_librosa.__spec__ = ModuleSpec("librosa", loader=None)
            fake_filters.__spec__ = ModuleSpec("librosa.filters", loader=None)
            fake_filters.mel = lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("Mel filter generation is not used by TTS inference")
            )
            fake_librosa.filters = fake_filters
            fake_librosa.util = types.SimpleNamespace(pad_center=lambda data, **_kwargs: data)
            sys.modules["librosa"] = fake_librosa
            sys.modules["librosa.filters"] = fake_filters

    @staticmethod
    def _force_g2pkk_mecab() -> None:
        """Use python-mecab-ko instead of the Windows-only eunjeon binary."""
        if platform.system() != "Windows":
            return
        import g2pkk.g2pkk as g2pkk_module
        from MeCab import MeCab

        g2pkk_module.G2p.check_mecab = lambda _self: None
        g2pkk_module.G2p.get_mecab = lambda _self: MeCab()

    def synthesize(self, text: str) -> tuple[str, Path, float]:
        normalized = " ".join(text.split())
        if not normalized:
            raise ValueError("텍스트를 입력해 주세요.")
        if len(normalized) > 500:
            raise ValueError("테스트 입력은 500자 이하로 제한됩니다.")

        output_dir = self._settings.secretary_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_id = uuid.uuid4().hex
        destination = output_dir / f"{audio_id}.wav"
        temporary = output_dir / f".{audio_id}.tmp.wav"

        with self._lock:
            model = self._get_model()
            speaker_id = model.hps.data.spk2id["KR"]
            model.tts_to_file(normalized, speaker_id, str(temporary), speed=1.0)

        with wave.open(str(temporary), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
            if audio.getnchannels() != 1 or duration < 0.2:
                temporary.unlink(missing_ok=True)
                raise RuntimeError("유효하지 않은 TTS WAV가 생성되었습니다.")
        temporary.replace(destination)
        return audio_id, destination, duration
