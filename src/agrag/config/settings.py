"""Configuration settings for the Agentic GraphRAG system."""

import os
from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "agrag"
    app_version: str = "0.1.0"
    debug: bool = False

    # LangSmith Observability
    langchain_tracing_v2: bool = True
    langchain_api_key: Optional[str] = None
    langchain_project: str = "agrag-test-scope-analysis"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # Google Generative AI
    google_api_key: Optional[str] = None
    google_model: str = "gemini-2.0-flash-exp"
    google_embedding_model: str = "models/text-embedding-004"
    google_thinking_level: Optional[str] = None
    google_thinking_budget: Optional[int] = None
    embedding_dimensions: int = 768
    llm_timeout_seconds: int = 45

    # Neo4j Configuration
    neo4j_uri: Optional[str] = None
    neo4j_username: str = "neo4j"
    neo4j_password: Optional[str] = None
    neo4j_database: str = "neo4j"

    # PostgreSQL Configuration
    postgres_host: Optional[str] = None
    postgres_port: int = 5432
    postgres_database: str = "agrag"
    postgres_user: Optional[str] = None
    postgres_password: Optional[str] = None
    postgres_sslmode: str = "require"

    # Neon PostgreSQL (alternative to generic PostgreSQL)
    neon_connection_string: Optional[str] = None

    # Agent Configuration
    max_tool_calls: int = 10
    max_model_calls: int = 20
    max_iterations: int = 15
    agent_temperature: float = 0.0

    # Retrieval Configuration
    default_retrieval_k: int = 10
    vector_search_similarity_threshold: float = 0.7
    graph_traversal_max_depth: int = 3
    hybrid_rrf_k: int = 60  # Reciprocal Rank Fusion constant

    # Middleware Configuration
    enable_pii_detection: bool = True
    enable_context_compaction: bool = True
    context_compaction_threshold: int = 4000  # tokens
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0  # seconds

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "text"
    log_file: Optional[str] = None

    @property
    def postgres_connection_string(self) -> str:
        """Build PostgreSQL connection string from components."""
        if self.neon_connection_string:
            return self.neon_connection_string

        if not all([self.postgres_host, self.postgres_user, self.postgres_password]):
            raise ValueError("PostgreSQL connection parameters are not fully configured")

        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
            f"?sslmode={self.postgres_sslmode}"
        )

    def validate_llm_config(self) -> None:
        """Validate LLM configuration."""
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY must be set")

    def validate_database_config(self) -> None:
        """Validate database configuration."""
        if not self.neo4j_uri or not self.neo4j_password:
            raise ValueError("Neo4j configuration (URI, password) must be set")

        if not self.neon_connection_string and not all(
            [self.postgres_host, self.postgres_user, self.postgres_password]
        ):
            raise ValueError(
                "PostgreSQL configuration must be set (either NEON_CONNECTION_STRING or individual parameters)"
            )

    def validate_langsmith_config(self) -> None:
        """Validate LangSmith configuration."""
        if self.langchain_tracing_v2 and not self.langchain_api_key:
            raise ValueError(
                "LANGCHAIN_API_KEY must be set when LangSmith tracing is enabled"
            )

    @field_validator("google_thinking_level")
    @classmethod
    def validate_thinking_level(cls, value: Optional[str]) -> Optional[str]:
        """Ensure thinking level is one of the supported Gemini values."""
        if value is None:
            return value

        normalized = value.lower()
        if normalized not in {"low", "high"}:
            raise ValueError("GOOGLE_THINKING_LEVEL must be either 'low' or 'high'")
        return normalized

    @field_validator("google_thinking_budget")
    @classmethod
    def validate_thinking_budget(cls, value: Optional[int]) -> Optional[int]:
        """Ensure thinking budget is within supported bounds."""
        if value is None:
            return value

        if value < -1:
            raise ValueError("GOOGLE_THINKING_BUDGET must be -1 (dynamic) or non-negative")

        return value


# Global settings instance
settings = Settings()


def setup_langsmith() -> None:
    """Configure LangSmith environment variables for tracing."""
    if settings.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if settings.langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
