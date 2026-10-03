from functools import lru_cache

from app.ai.providers.ollama import OllamaProvider
from app.ai.providers.openai import OpenAIProvider
from app.ai.router import LLMRouter
from app.core.config import get_settings


@lru_cache
def get_llm_router() -> LLMRouter:
    """
    Construye y cachea el router de proveedores LLM.

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

    providers = {
        "ollama": OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=120.0,
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

    fallback_provider = settings.fallback_llm_provider

    # Si el fallback configurado no está disponible en este entorno,
    # lo deshabilitamos en lugar de construir un router inconsistente.
    if (
        fallback_provider is not None
        and fallback_provider not in providers
    ):
        fallback_provider = None

    return LLMRouter(
        providers=providers,
        primary_provider=settings.primary_llm_provider,
        fallback_provider=fallback_provider,
        max_retries=settings.llm_max_retries,
        retry_delay_seconds=settings.llm_retry_delay_seconds,
    )