from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from pydantic import BaseModel

class DatabaseConfig(BaseModel):
    url: str
    async_url: str
    pool_size: int
    pool_timeout: int
    max_overflow: int
    sql_echo: bool

class ElasticsearchConfig(BaseModel):
    url: str
    index: str
    username: str
    password: str

class PaperLoaderConfig(BaseModel):
    output_dir: str
    batch_size: int
    workers: int
    use_postgres: bool
    arxiv_metadata_path: str
    process_pdfs: bool = False  # Default to False for fast loading
    
class QdrantConfig(BaseModel):
    url: str
    vector_size: int
    collection: str
    distance: str
    output_dir: str

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="forbid",
        validate_assignment=True,
        case_sensitive=False
    )
    # LangSmith configuration (optional)
    LANGSMITH_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "ai-researcher-proto"
    
    OPENAI_API_KEY: str
    MODEL_NAME: str

    DATABASE_URL: str
    DATABASE_ASYNC_URL: str
    DATABASE_POOL_SIZE: int
    DATABASE_POOL_TIMEOUT: int
    DATABASE_MAX_OVERFLOW: int
    DATABASE_SQL_ECHO: bool
    
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    ELASTICSEARCH_URL: str
    ELASTICSEARCH_INDEX: str
    ELASTICSEARCH_USERNAME: str
    ELASTICSEARCH_PASSWORD: str

    LOADER_OUTPUT_DIR: str
    LOADER_BATCH_SIZE: int
    LOADER_WORKERS: int
    LOADER_USE_POSTGRES: bool
    LOADER_ARXIV_METADATA_PATH: str
    LOADER_PROCESS_PDFS: bool = False

    QDRANT_URL: str
    QDRANT_VECTOR_SIZE: int
    QDRANT_COLLECTION: str
    QDRANT_DISTANCE: str

    @property 
    def database_config(self) -> DatabaseConfig:
        return DatabaseConfig(
            url=self.DATABASE_URL,
            async_url=self.DATABASE_ASYNC_URL,
            pool_size=self.DATABASE_POOL_SIZE,
            pool_timeout=self.DATABASE_POOL_TIMEOUT,
            max_overflow=self.DATABASE_MAX_OVERFLOW,
            sql_echo=self.DATABASE_SQL_ECHO
        )
    
    @property
    def elasticsearch_config(self) -> ElasticsearchConfig:
        return ElasticsearchConfig(
            url=self.ELASTICSEARCH_URL,
            index=self.ELASTICSEARCH_INDEX,
            username=self.ELASTICSEARCH_USERNAME,
            password=self.ELASTICSEARCH_PASSWORD
        )
    
    @property
    def paper_loader_config(self) -> PaperLoaderConfig:
        return PaperLoaderConfig(
            output_dir=self.LOADER_OUTPUT_DIR,
            batch_size=self.LOADER_BATCH_SIZE,
            workers=self.LOADER_WORKERS,
            use_postgres=self.LOADER_USE_POSTGRES,
            arxiv_metadata_path=self.LOADER_ARXIV_METADATA_PATH,
            process_pdfs=self.LOADER_PROCESS_PDFS
        )

    @property
    def qdrant_config(self) -> QdrantConfig:
        return QdrantConfig(
            url=self.QDRANT_URL,
            vector_size=self.QDRANT_VECTOR_SIZE,
            collection=self.QDRANT_COLLECTION,
            distance=self.QDRANT_DISTANCE,
            output_dir=self.LOADER_OUTPUT_DIR,
        )

settings = Settings()
