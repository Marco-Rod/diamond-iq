from unittest.mock import AsyncMock

import pytest

from app.ai.providers.exceptions import (
    LLMProviderPermanentError,
    LLMProviderTimeoutError,
)
from app.ai.router import (
    LLMProviderNotFoundError,
    LLMRouter,
)
from app.ai.schemas import (
    LLMProviderState,
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

@pytest.mark.anyio
async def test_router_retries_transient_provider_error() -> None:
    provider = AsyncMock()

    provider.generate.side_effect = [
        LLMProviderTimeoutError(
            "Provider timed out."
        ),
        LLMResponse(
            content="Recovered.",
            provider="ollama",
            model="qwen3:8b",
        ),
    ]

    router = LLMRouter(
        providers={
            "ollama": provider,
        },
        primary_provider="ollama",
        max_retries=1,
        retry_delay_seconds=0,
    )

    response = await router.generate(
        LLMRequest(
            messages=[
                {
                    "role": "user",
                    "content": "Hello",
                }
            ]
        )
    )

    assert response.content == "Recovered."

    assert provider.generate.await_count == 2

@pytest.mark.anyio
async def test_router_does_not_retry_permanent_error() -> None:
    provider = AsyncMock()

    provider.generate.side_effect = (
        LLMProviderPermanentError(
            "Invalid model."
        )
    )

    router = LLMRouter(
        providers={
            "ollama": provider,
        },
        primary_provider="ollama",
        max_retries=3,
        retry_delay_seconds=0,
    )

    with pytest.raises(
        LLMProviderPermanentError
    ):
        await router.generate(
            LLMRequest(
                messages=[
                    {
                        "role": "user",
                        "content": "Hello",
                    }
                ]
            )
        )

    # Aunque configuramos 3 retries,
    # un error permanente solo se intenta una vez.
    assert provider.generate.await_count == 1

@pytest.mark.anyio
async def test_router_falls_back_after_transient_failures() -> None:
    primary = AsyncMock()
    fallback = AsyncMock()

    primary.generate.side_effect = (
        LLMProviderTimeoutError(
            "Ollama timed out."
        )
    )

    fallback.generate.return_value = LLMResponse(
        content="Fallback response.",
        provider="fallback",
        model="fallback-model",
    )

    router = LLMRouter(
        providers={
            "ollama": primary,
            "fallback": fallback,
        },
        primary_provider="ollama",
        fallback_provider="fallback",
        max_retries=1,
        retry_delay_seconds=0,
    )

    response = await router.generate(
        LLMRequest(
            messages=[
                {
                    "role": "user",
                    "content": "Hello",
                }
            ]
        )
    )

    assert response.provider == "fallback"

    # Intento original + retry.
    assert primary.generate.await_count == 2

    # Después usamos fallback.
    fallback.generate.assert_awaited_once()

@pytest.mark.anyio
async def test_explicit_provider_does_not_use_fallback() -> None:
    primary = AsyncMock()
    fallback = AsyncMock()

    primary.generate.side_effect = (
        LLMProviderTimeoutError(
            "Ollama timed out."
        )
    )

    router = LLMRouter(
        providers={
            "ollama": primary,
            "fallback": fallback,
        },
        primary_provider="ollama",
        fallback_provider="fallback",
        max_retries=0,
        retry_delay_seconds=0,
    )

    with pytest.raises(
        LLMProviderTimeoutError
    ):
        await router.generate(
            LLMRequest(
                messages=[
                    {
                        "role": "user",
                        "content": "Hello",
                    }
                ]
            ),
            provider_name="ollama",
        )

    fallback.generate.assert_not_awaited()


@pytest.mark.anyio
async def test_router_keeps_provider_that_owns_state() -> None:
    """
    Verifica que una conversación iniciada por Gemini
    continúe utilizando Gemini aunque Ollama sea el
    provider primario.

    Esto es necesario porque el estado interno de Gemini
    no puede ser interpretado por otro provider.
    """

    ollama = AsyncMock()
    gemini = AsyncMock()

    gemini.generate.return_value = (
        LLMResponse(
            content="Continued.",
            provider="gemini",
            model=(
                "gemini-3.8-flash"
            ),
        )
    )

    router = LLMRouter(
        providers={
            "ollama": ollama,
            "gemini": gemini,
        },
        primary_provider="ollama",
        fallback_provider="gemini",
        max_retries=0,
        retry_delay_seconds=0,
    )

    request = LLMRequest(
        messages=[
            {
                "role": "user",
                "content": (
                    "Continue."
                ),
            }
        ],
        tools=[],
        provider_state=(
            LLMProviderState(
                provider="gemini",
                data={
                    "interaction_id": (
                        "interaction-123"
                    ),
                },
            )
        ),
    )

    response = await router.generate(
        request,
    )

    assert (
        response.provider
        == "gemini"
    )

    # Ollama es el provider primario, pero no debe participar
    # porque Gemini es propietario del estado actual.
    ollama.generate.assert_not_awaited()

    gemini.generate.assert_awaited_once_with(
        request
    )

@pytest.mark.anyio
async def test_router_rejects_explicit_provider_that_does_not_own_state() -> None:
    """
    Evita continuar una conversación utilizando un provider
    diferente al que creó el estado actual.
    """

    ollama = AsyncMock()
    gemini = AsyncMock()

    router = LLMRouter(
        providers={
            "ollama": ollama,
            "gemini": gemini,
        },
        primary_provider="ollama",
        fallback_provider="gemini",
        max_retries=0,
        retry_delay_seconds=0,
    )

    request = LLMRequest(
        messages=[
            {
                "role": "user",
                "content": "Continue.",
            }
        ],
        tools=[],
        provider_state=(
            LLMProviderState(
                provider="gemini",
                data={
                    "interaction_id": (
                        "interaction-123"
                    ),
                },
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Explicit provider does not match"
        ),
    ):
        await router.generate(
            request,
            provider_name="ollama",
        )

    ollama.generate.assert_not_awaited()
    gemini.generate.assert_not_awaited()