from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    secretary_model_provider: str = "local"
    secretary_output_dir: Path = Path("data/output")
    secretary_wsl_distro: str = "Ubuntu"


settings = Settings()
