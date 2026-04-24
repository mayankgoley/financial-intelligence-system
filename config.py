from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(description="embeddings")
    ANTHROPIC_API_KEY: str = Field(default="", description="claude agents")

    LANGCHAIN_TRACING_V2: bool = Field(default=True)
    LANGCHAIN_API_KEY: str = Field(default="")
    LANGCHAIN_PROJECT: str = Field(default="financial-intelligence-system")

    POSTGRES_USER: str = Field(default="finai")
    POSTGRES_PASSWORD: str = Field(default="finai_password_change_me")
    POSTGRES_DB: str = Field(default="financial_intelligence")
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)

    @property
    def DATABASE_URL(self) -> str:
        return (f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
                f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}")

    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)

    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_COLLECTION_NAME: str = Field(default="financial_chunks")

    KAFKA_BOOTSTRAP_SERVERS: str = Field(default="localhost:9092")

    EMBEDDING_MODEL: str = Field(default="text-embedding-3-large")
    EMBEDDING_DIMENSIONS: int = Field(default=3072)

    CHUNK_SIZE: int = Field(default=512)
    CHUNK_OVERLAP: int = Field(default=64)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
