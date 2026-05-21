from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="DicomVision Server", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    cors_origins: list[str] = Field(default=["*"], alias="CORS_ORIGINS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    web_sample_dicom_path: str | None = Field(default=None, alias="WEB_SAMPLE_DICOM_PATH")
    web_upload_dicom_root: str | None = Field(default=None, alias="WEB_UPLOAD_DICOM_ROOT")
    web_upload_max_files: int = Field(default=5000, alias="WEB_UPLOAD_MAX_FILES")
    web_upload_max_bytes: int = Field(default=2 * 1024 * 1024 * 1024, alias="WEB_UPLOAD_MAX_BYTES")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
