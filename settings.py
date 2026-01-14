"""Settings configuration for Homecare Memory."""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file in the same directory as this file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Database Configuration
    database_url: str = Field(
        ...,
        description="PostgreSQL connection URL with asyncpg"
    )

    # OpenAI Configuration
    openai_api_key: str = Field(
        ...,
        description="API key for OpenAI"
    )

    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model to use"
    )

    embedding_dimension: int = Field(
        default=1536,
        description="Embedding vector dimension"
    )

    # Vector Store Configuration
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="Directory for Chroma persistent storage"
    )

    # Connection Pool Configuration
    db_pool_min_size: int = Field(
        default=5,
        description="Minimum database connection pool size"
    )

    db_pool_max_size: int = Field(
        default=20,
        description="Maximum database connection pool size"
    )

    # Performance Configuration
    default_search_limit: int = Field(
        default=3,
        description="Default number of search results (reduced from mem0's 10)"
    )

    max_search_limit: int = Field(
        default=10,
        description="Maximum number of search results allowed"
    )


def load_settings() -> Settings:
    """Load settings with proper error handling."""
    try:
        return Settings()
    except Exception as e:
        error_msg = f"Failed to load settings: {e}"
        if "database_url" in str(e).lower():
            error_msg += "\nMake sure to set DATABASE_URL in your .env file"
        if "openai_api_key" in str(e).lower():
            error_msg += "\nMake sure to set OPENAI_API_KEY in your .env file"
        raise ValueError(error_msg) from e
