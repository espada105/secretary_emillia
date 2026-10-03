from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re
import wave

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .config import settings
from .enums import TTSEngine, VoiceName
from .expressive_tts import ExpressiveTTS, RVC_MODELS
from .tts import KoreanTTS

app = FastAPI(title="Secretary Emillia — Local TTS", docs_url="/docs")
tts = KoreanTTS(settings)
expressive_tts = ExpressiveTTS(settings)
PAGE = Path(__file__).resolve().parents[2] / "static" / "index.html"


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    engine: TTSEngine = TTSEngine.QWEN3
    voice: VoiceName = VoiceName.EMILIA
    tone: str = Field(default="bright", min_length=1, max_length=24, pattern=r"[A-Za-z0-9_-]+")
    instruct: str = Field(
        default=(
            "Speak as a bright, playful anime character. "
            "Use lively pitch variation and clear Korean pronunciation."
        ),
        max_length=500,
    )
    rvc_index_rate: float = Field(default=0.45, ge=0.0, le=1.0)
    rvc_protect: float = Field(default=0.45, ge=0.0, le=0.5)
    rvc_pitch: int = Field(default=1, ge=-12, le=12)


class TTSResponse(BaseModel):
    audio_id: str
    audio_url: str
    duration_seconds: float
    voice: VoiceName
    title: str


class AudioHistoryItem(BaseModel):
    audio_id: str
    audio_url: str
    duration_seconds: float
    created_at: datetime
    title: str


SAFE_AUDIO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*")
TITLED_AUDIO_ID = re.compile(
    r"(?:(?P<engine>qwen3(?:_korean)?)-)?(?P<voice>[a-z_]+)(?:-tone(?P<tone>[a-z0-9_-]+))?-pitch(?P<pitch>[+-]\d+)-strength(?P<strength>\d\.\d{2})"
    r"-protect(?P<protect>\d\.\d{2})-\d{8}-\d{6}-[0-9a-f]{8}"
)


def audio_title(audio_id: str) -> str:
    match = TITLED_AUDIO_ID.fullmatch(audio_id)
    if match is None:
        return audio_id
    values = match.groupdict()
    voice = values["voice"].replace("_", " ").title()
    engine = {"qwen3": "Qwen3 1.7B", "qwen3_korean": "Qwen3 0.6B Korean"}.get(values["engine"], "Qwen")
    tone = {"bright": "밝음", "calm": "차분함", "excited": "흥분", "serious": "진지함", "natural": "기본", "custom": "직접 지시"}.get(values["tone"], values["tone"] or "미지정")
    return (
        f"{engine} · {voice} · 톤 {tone} · 피치 {values['pitch']} · 음색 강도 {values['strength']}"
        f" · 자음 보호 {values['protect']}"
    )


@app.get("/", response_class=HTMLResponse)
def test_page() -> str:
    return PAGE.read_text(encoding="utf-8")


@app.get("/api/voices")
def voices() -> dict[str, list[str]]:
    return {
        "available": [VoiceName.BASE_KOREAN, *RVC_MODELS],
        "planned_rvc": [voice for voice in VoiceName if voice not in {VoiceName.BASE_KOREAN, *RVC_MODELS}],
    }


@app.post("/api/tts", response_model=TTSResponse)
def synthesize(request: TTSRequest) -> TTSResponse:
    try:
        if request.engine == TTSEngine.MELO:
            if request.voice != VoiceName.BASE_KOREAN:
                raise ValueError("Melo 엔진은 기본 한국어 음성만 지원합니다. Qwen3을 선택해 주세요.")
            audio_id, _path, duration = tts.synthesize(request.text)
        else:
            audio_id, _path, duration = expressive_tts.synthesize(
                text=request.text,
                engine=request.engine,
                voice=request.voice,
                tone=request.tone,
                instruct=request.instruct,
                index_rate=request.rvc_index_rate,
                protect=request.rvc_protect,
                pitch=request.rvc_pitch,
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS 생성 실패: {exc}") from exc
    return TTSResponse(
        audio_id=audio_id,
        audio_url=f"/api/audio/{audio_id}",
        duration_seconds=round(duration, 2),
        voice=request.voice,
        title=audio_title(audio_id),
    )


@app.get("/api/audio", response_model=list[AudioHistoryItem])
def audio_history() -> list[AudioHistoryItem]:
    output_dir = settings.secretary_output_dir
    if not output_dir.is_dir():
        return []
    history: list[AudioHistoryItem] = []
    for path in sorted(output_dir.glob("*.wav"), key=lambda item: item.stat().st_mtime, reverse=True):
        if not SAFE_AUDIO_ID.fullmatch(path.stem):
            continue
        try:
            with wave.open(str(path), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
        except (EOFError, wave.Error):
            continue
        history.append(
            AudioHistoryItem(
                audio_id=path.stem,
                audio_url=f"/api/audio/{path.stem}",
                duration_seconds=round(duration, 2),
                created_at=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
                title=audio_title(path.stem),
            )
        )
    return history[:50]


@app.get("/api/audio/{audio_id}")
def audio(audio_id: str) -> FileResponse:
    if not SAFE_AUDIO_ID.fullmatch(audio_id):
        raise HTTPException(status_code=404, detail="오디오를 찾을 수 없습니다.")
    path = settings.secretary_output_dir / f"{audio_id}.wav"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="오디오를 찾을 수 없습니다.")
    return FileResponse(path, media_type="audio/wav", filename="secretary-tts.wav")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "local-only"}
