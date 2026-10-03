from enum import StrEnum


class AgentName(StrEnum):
    ORCHESTRATOR = "orchestrator"
    VOICE = "voice"


class ToolName(StrEnum):
    KOREAN_TTS = "korean_tts"
    RVC_CONVERT = "rvc_convert"


class VoiceName(StrEnum):
    BASE_KOREAN = "base_korean"
    ONO_ANNA = "ono_anna"
    EMILIA = "emilia"
    RAM = "ram"
    BEATRICE = "beatrice"
    LILAC = "lilac"
    SHION = "shion"
    TISERA = "tisera"


class TTSEngine(StrEnum):
    MELO = "melo"
    QWEN3 = "qwen3"
    QWEN3_KOREAN = "qwen3_korean"


class EnvKey(StrEnum):
    SECRETARY_MODEL_PROVIDER = "SECRETARY_MODEL_PROVIDER"
    OPENAI_API_KEY = "OPENAI_API_KEY"
    ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
    SECRETARY_OUTPUT_DIR = "SECRETARY_OUTPUT_DIR"
    HF_HOME = "HF_HOME"
    SECRETARY_WSL_DISTRO = "SECRETARY_WSL_DISTRO"
