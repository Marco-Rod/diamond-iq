from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DiamondIQ"
    database_url: str

    ai_max_tool_iterations: int = 3
    ai_max_tool_calls_per_chat: int = 6
    ai_max_total_tokens_per_chat: int = 12_000

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"

    openai_api_key: str | None = None
    openai_model: str = "gpt-6-luna"

    primary_llm_provider: str = "ollama"
    fallback_llm_provider: str | None = "openai"

    llm_max_retries: int = 1
    llm_retry_delay_seconds: float = 0.5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings()  -> Settings:
    return Settings()
