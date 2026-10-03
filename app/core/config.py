from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DiamondIQ"
    database_url: str

    ai_max_tool_iterations: int = 3
    ai_max_tool_calls_per_chat: int = 6
    ai_max_total_tokens_per_chat: int = 12_000

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_timeout_seconds: float = 30.0
    
    openai_api_key: str | None = None
    openai_model: str = "gpt-6-luna"

    # Modelo utilizado para convertir texto en vectores durante
    # el proceso de recuperación de conocimiento.
    ollama_embedding_model: str = "nomic-embed-text"

    # Timeout específico para generación de embeddings.
    # Puede ser mayor que el timeout de chat porque la primera carga
    # del modelo de embeddings puede tardar varios segundos.
    ollama_embedding_timeout_seconds: float = 30.0

    # Configuración de Gemini.
    #
    # La API key es opcional para que Diamond IQ pueda seguir
    # funcionando únicamente con Ollama en desarrollo local.
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.8-flash"
    gemini_timeout_seconds: float = 30.0

    # Provider principal utilizado cuando una conversación
    # todavía no pertenece a ningún provider específico.
    primary_llm_provider: str = "ollama"

    # Provider alternativo utilizado cuando el principal falla
    # con un error clasificado como transitorio.
    fallback_llm_provider: str | None = None
    
    @field_validator(
        "fallback_llm_provider",
        mode="before",
    )
    @classmethod
    def normalize_optional_provider(
        cls,
        value: str | None,
    ) -> str | None:
        """
        Convierte valores vacíos de configuración a None.

        Esto permite usar en .env:

            FALLBACK_LLM_PROVIDER=

        sin que el router intente buscar un provider llamado "".
        """

        if value is None:
            return None

        if isinstance(value, str):
            value = value.strip()

            if not value:
                return None

        return value
    
    # Política de resiliencia común para los providers.
    llm_max_retries: int = 1
    llm_retry_delay_seconds: float = 0.5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings()  -> Settings:
    return Settings()
