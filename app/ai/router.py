from collections.abc import Mapping

from app.ai.providers.base import LLMProvider
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

        self._validate_provider(primary_provider)

        if fallback_provider is not None:
            self._validate_provider(fallback_provider)

    async def generate(
        self,
        request: LLMRequest,
        *,
        provider_name: str | None = None,
    ) -> LLMResponse:
        """
        Envía una petición al provider seleccionado.

        Si no se especifica `provider_name`, utilizamos el provider
        principal configurado para Diamond IQ.

        En esta primera versión todavía no ejecutamos fallback automático.
        Esa responsabilidad se añadirá explícitamente después para poder
        controlar qué tipos de error justifican un fallback.
        """

        selected_provider = (
            provider_name or self.primary_provider
        )

        provider = self.get_provider(selected_provider)

        return await provider.generate(request)

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