from functools import lru_cache

from app.ai.providers.base import LLMProvider
from app.ai.providers.gemini import GeminiProvider
from app.ai.providers.ollama import OllamaProvider
from app.ai.providers.openai import OpenAIProvider
from app.ai.router import LLMRouter
from app.core.config import get_settings


@lru_cache
def get_llm_router() -> LLMRouter:
    """
    Construye y cachea el router de proveedores LLM.
    Construye el router de LLMs disponibles para Diamond IQ.

    Ollama siempre está disponible como provider local.

    Gemini se registra únicamente cuando existe una API key
    configurada. Esto permite ejecutar el proyecto completamente
    en local sin depender obligatoriamente de servicios externos.

    Esta función concentra la creación de infraestructura relacionada
    con IA para evitar que endpoints o services tengan que conocer:

    - URLs de proveedores;
    - API keys;
    - modelos concretos;
    - provider primario;
    - provider de fallback.

    La configuración proviene de Settings, por lo que el comportamiento
    puede cambiar entre entornos sin modificar el código.
    """

    settings = get_settings()

    providers: dict[str, LLMProvider] = {
        "ollama": OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=(
                settings.ollama_timeout_seconds
            ),
        ),
    }

    # OpenAI solo se registra si existe una API key.
    #
    # Esto permite desarrollar localmente únicamente con Ollama sin que
    # la aplicación falle durante el arranque por no tener credenciales.
    if settings.openai_api_key:
        providers["openai"] = OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )

    # Gemini es opcional.
    #
    # Solo lo registramos cuando existe una API key para evitar
    # construir un provider cloud inutilizable en entornos locales.
    if settings.gemini_api_key:
        providers["gemini"] = GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            timeout_seconds=(
                settings.gemini_timeout_seconds
            ),
        )

    primary_provider = (
        settings.primary_llm_provider
    )

    fallback_provider = (
        settings.fallback_llm_provider
    )

    if primary_provider not in providers:
            raise ValueError(
                "The configured primary LLM provider "
                f"'{primary_provider}' is not available."
            )

    if (
        fallback_provider
        and fallback_provider not in providers
    ):
        raise ValueError(
            "The configured fallback LLM provider "
            f"'{fallback_provider}' is not available."
        )
    # Si el fallback configurado no está disponible en este entorno,
    # lo deshabilitamos en lugar de construir un router inconsistente.

    return LLMRouter(
        providers=providers,
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
        max_retries=(
            settings.llm_max_retries
        ),
        retry_delay_seconds=(
            settings.llm_retry_delay_seconds
        ),
    )