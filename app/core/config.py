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
    expose_api_docs: bool | None = Field(default=None, alias="EXPOSE_API_DOCS")
    web_sample_dicom_path: str | None = Field(default=None, alias="WEB_SAMPLE_DICOM_PATH")
    web_upload_dicom_root: str | None = Field(default=None, alias="WEB_UPLOAD_DICOM_ROOT")
    web_upload_max_files: int = Field(default=5000, alias="WEB_UPLOAD_MAX_FILES")
    web_upload_max_bytes: int = Field(default=2 * 1024 * 1024 * 1024, alias="WEB_UPLOAD_MAX_BYTES")
    web_upload_max_age_seconds: int = Field(default=30 * 60, alias="WEB_UPLOAD_MAX_AGE_SECONDS")
    web_upload_cleanup_interval_seconds: int = Field(
        default=30 * 60,
        alias="WEB_UPLOAD_CLEANUP_INTERVAL_SECONDS",
    )
    pacs_wado_cache_root: str | None = Field(default=None, alias="PACS_WADO_CACHE_ROOT")
    pacs_wado_cache_max_age_seconds: int = Field(default=24 * 60 * 60, alias="PACS_WADO_CACHE_MAX_AGE_SECONDS")
    pacs_wado_cache_cleanup_interval_seconds: int = Field(
        default=60 * 60,
        alias="PACS_WADO_CACHE_CLEANUP_INTERVAL_SECONDS",
    )

    @property
    def api_docs_enabled(self) -> bool:
        if self.expose_api_docs is not None:
            return self.expose_api_docs
        return self.app_env.lower() not in {"prod", "production"}

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
