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

    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/ai_researcher"
    DATABASE_ASYNC_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/ai_researcher"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_MAX_OVERFLOW: int = 20
    SQL_ECHO: bool = False
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "ai_researcher"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"

    WEAVIATE_URL: str = ""
    WEAVIATE_API_KEY: str = ""

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Optional: LangSmith settings for debugging and monitoring
    LANGSMITH_API_KEY: str = ""
settings = Settings()
