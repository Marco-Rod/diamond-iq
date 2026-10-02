from unittest.mock import AsyncMock

import pytest

from app.ai.router import (
    LLMProviderNotFoundError,
    LLMRouter,
)
from app.ai.schemas import (
    LLMRequest,
    LLMResponse,
)


def build_request() -> LLMRequest:
    """
    Construye un request mínimo reutilizable para los tests.
    """

    return LLMRequest(
        messages=[
            {
                "role": "user",
                "content": "Who has the highest Power?",
            }
        ]
    )


@pytest.mark.anyio
async def test_router_uses_primary_provider_by_default() -> None:
    """
    Si no especificamos provider, el router debe utilizar
    el provider principal.
    """

    ollama = AsyncMock()

    ollama.generate.return_value = LLMResponse(
        content="Response from Ollama.",
        provider="ollama",
        model="qwen2.5-coder:7b",
    )

    router = LLMRouter(
        providers={
            "ollama": ollama,
        },
        primary_provider="ollama",
    )

    response = await router.generate(
        build_request(),
    )

    assert response.provider == "ollama"

    ollama.generate.assert_awaited_once()


@pytest.mark.anyio
async def test_router_can_select_specific_provider() -> None:
    """
    El caller puede solicitar explícitamente un provider registrado.
    """

    ollama = AsyncMock()
    openai = AsyncMock()

    openai.generate.return_value = LLMResponse(
        content="Response from OpenAI.",
        provider="openai",
        model="test-model",
    )

    router = LLMRouter(
        providers={
            "ollama": ollama,
            "openai": openai,
        },
        primary_provider="ollama",
        fallback_provider="openai",
    )

    response = await router.generate(
        build_request(),
        provider_name="openai",
    )

    assert response.provider == "openai"

    openai.generate.assert_awaited_once()
    ollama.generate.assert_not_awaited()


def test_router_rejects_unknown_primary_provider() -> None:
    """
    Una configuración inválida debe fallar inmediatamente
    durante la construcción del router.
    """

    with pytest.raises(LLMProviderNotFoundError):
        LLMRouter(
            providers={},
            primary_provider="unknown",
        )


@pytest.mark.anyio
async def test_router_rejects_unknown_requested_provider() -> None:
    """
    Tampoco permitimos seleccionar providers no registrados
    durante una petición.
    """

    ollama = AsyncMock()

    router = LLMRouter(
        providers={
            "ollama": ollama,
        },
        primary_provider="ollama",
    )

    with pytest.raises(LLMProviderNotFoundError):
        await router.generate(
            build_request(),
            provider_name="unknown",
        )