from __future__ import annotations

import httpx

from app.ai.providers.exceptions import (
    LLMProviderTimeoutError,
    LLMProviderUnavailableError,
)


class OllamaEmbeddingProvider:
    """
    Cliente responsable exclusivamente de generar embeddings
    utilizando Ollama.

    Separamos embeddings de OllamaProvider porque son dos
    responsabilidades diferentes:

    - OllamaProvider genera respuestas de lenguaje.
    - OllamaEmbeddingProvider convierte texto en vectores.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 8.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Convierte uno o varios textos en vectores numéricos.
        """

        if not texts:
            return []

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
            ) as client:
                response = await client.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": self.model,
                        "input": texts,
                    },
                )

                response.raise_for_status()

        except httpx.TimeoutException as exc:
            raise LLMProviderTimeoutError(
                "Ollama embedding request timed out."
            ) from exc

        except httpx.RequestError as exc:
            raise LLMProviderUnavailableError(
                "Could not connect to Ollama embeddings."
            ) from exc

        data = response.json()

        return data["embeddings"]