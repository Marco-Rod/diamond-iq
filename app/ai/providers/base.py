from typing import Protocol

from app.ai.schemas import LLMRequest, LLMResponse


class LLMProvider(Protocol):
    """
    Contrato común que debe cumplir cualquier proveedor LLM.

    AIService y LLMRouter dependerán de esta abstracción en lugar de
    depender directamente de Ollama, OpenAI u otro proveedor concreto.

    Esto aplica Dependency Inversion: las capas superiores dependen
    de una interfaz estable y los proveedores adaptan su SDK/API
    a ese contrato.
    """

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Genera una respuesta utilizando el proveedor.

        Cada implementación será responsable de traducir `LLMRequest`
        al formato requerido por su API y posteriormente normalizar
        la respuesta a `LLMResponse`.
        """
        ...