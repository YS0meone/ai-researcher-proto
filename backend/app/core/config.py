from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="forbid",
        validate_assignment=True,
        case_sensitive=False
    )

    OPENAI_API_KEY: str

    DATABASE_URL: str
    DATABASE_POOL_SIZE: int
    DATABASE_POOL_TIMEOUT: int
    DATABASE_MAX_OVERFLOW: int
    SQL_ECHO: bool
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    WEAVIATE_URL: str
    WEAVIATE_API_KEY: str

settings = Settings()
