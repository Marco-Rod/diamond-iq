import asyncio
from collections.abc import Mapping

from app.ai.providers.base import LLMProvider
from app.ai.providers.exceptions import (
    LLMProviderTransientError,
)
from app.ai.schemas import LLMRequest, LLMResponse


class LLMProviderNotFoundError(Exception):
    """
    Se lanza cuando el router recibe el nombre de un provider
    que no ha sido registrado.

    Utilizamos una excepción propia para evitar que las capas superiores
    tengan que interpretar KeyError u otros detalles internos.
    """


class LLMRouter:
    """
    Selecciona qué provider LLM debe procesar una petición.

    El router no conoce detalles específicos de Ollama ni OpenAI.
    Únicamente trabaja con objetos que cumplen el contrato LLMProvider.

    Esta separación permite cambiar providers mediante configuración
    sin modificar AIService ni los endpoints HTTP.
    """

    def __init__(
        self,
        *,
        providers: Mapping[str, LLMProvider],
        primary_provider: str,
        fallback_provider: str | None = None,
        max_retries: int = 1,
        retry_delay_seconds: float = 0.5,
    ) -> None:
        """
        Inicializa el router con un registro de providers.

        Ejemplo:

        {
            "ollama": OllamaProvider(...),
            "openai": OpenAIProvider(...),
        }

        `primary_provider` será utilizado por defecto.

        `fallback_provider` queda preparado para utilizarse cuando
        implementemos el comportamiento de resiliencia.
        """

        self.providers = dict(providers)

        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider

        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        self._validate_provider(primary_provider)

        if fallback_provider is not None:
            self._validate_provider(fallback_provider)

    async def generate(
    self,
    request: LLMRequest,
    provider_name: str | None = None,
    ) -> LLMResponse:
        selected_provider_name = (
            provider_name
            or self.primary_provider
        )

        provider = self.get_provider(
            selected_provider_name
        )

        try:
            return await self._generate_with_retry(
                provider=provider,
                request=request,
            )

        except LLMProviderTransientError:
            # Si el caller pidió explícitamente un provider,
            # respetamos esa decisión y no hacemos fallback automático.
            if provider_name is not None:
                raise

            # Sin fallback configurado no hay nada más que intentar.
            if self.fallback_provider is None:
                raise

            # Evitamos intentar exactamente el mismo provider otra vez.
            if (
                self.fallback_provider
                == selected_provider_name
            ):
                raise

            fallback = self.get_provider(
                self.fallback_provider
            )

            return await self._generate_with_retry(
                provider=fallback,
                request=request,
            )
    def get_provider(
        self,
        provider_name: str,
    ) -> LLMProvider:
        """
        Recupera un provider registrado por nombre.

        Centralizar esta búsqueda evita que distintas capas accedan
        directamente al diccionario interno.
        """

        try:
            return self.providers[provider_name]

        except KeyError:
            raise LLMProviderNotFoundError(
                f"LLM provider '{provider_name}' is not registered."
            ) from None

    def _validate_provider(
        self,
        provider_name: str,
    ) -> None:
        """
        Valida configuración durante el arranque.

        Es preferible detectar un provider mal configurado al construir
        el router y no esperar hasta recibir el primer request real.
        """

        if provider_name not in self.providers:
            raise LLMProviderNotFoundError(
                f"LLM provider '{provider_name}' is not registered."
            )

    async def _generate_with_retry(
    self,
    *,
    provider: LLMProvider,
    request: LLMRequest,
    ) -> LLMResponse:
        """
        Ejecuta una llamada al provider y reintenta únicamente errores
        clasificados explícitamente como transitorios.

        `max_retries=1` significa:
        - intento original;
        - un retry adicional.
        """

        for attempt in range(
            self.max_retries + 1
        ):
            try:
                return await provider.generate(
                    request,
                )

            except LLMProviderTransientError:
                # Ya consumimos todos los retries disponibles.
                if attempt >= self.max_retries:
                    raise

                await asyncio.sleep(
                    self.retry_delay_seconds
                )

        # Protección defensiva; normalmente nunca se alcanza.
        raise RuntimeError(
            "Provider retry loop ended unexpectedly."
        )