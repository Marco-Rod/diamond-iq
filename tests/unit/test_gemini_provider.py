from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.ai.providers.exceptions import (
    LLMProviderPermanentError,
    LLMProviderTransientError,
    LLMProviderUnavailableError,
)
from app.ai.providers.gemini import (
    GeminiProvider,
)
from app.ai.schemas import (
    LLMProviderState,
    LLMRequest,
)


class FakeGeminiAPIError(Exception):
    """
    Excepción ficticia utilizada para probar la traducción
    de códigos HTTP sin depender del constructor interno
    de google.genai.errors.APIError.
    """

    def __init__(self, code: int) -> None:
        super().__init__(
            f"Fake Gemini API error: {code}"
        )

        self.code = code

def build_provider() -> GeminiProvider:
    """
    Crea una instancia de GeminiProvider para los tests.

    Utilizamos una API key ficticia porque las llamadas reales
    al SDK se sustituyen mediante mocks.
    """

    return GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        timeout_seconds=5.0,
    )


def build_request(
    *,
    tools: list[dict] | None = None,
) -> LLMRequest:
    """
    Construye una petición representativa utilizada por
    distintos tests del provider.

    LLMRequest espera que `tools` sea una lista, por lo que
    utilizamos una lista vacía cuando no necesitamos tools.
    """

    return LLMRequest(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Diamond IQ Analyst."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Tell me about Ethan Carter."
                ),
            },
        ],
        temperature=0.2,
        tools=tools or [],
    )


@pytest.mark.anyio
async def test_gemini_provider_normalizes_text_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verifica que una respuesta de texto de Gemini se convierta
    correctamente a nuestro LLMResponse neutral.
    """

    provider = build_provider()

    interaction = SimpleNamespace(
        id="interaction-123",
        output_text=(
            "Ethan Carter plays for the Yankees."
        ),
        steps=[],
        usage=SimpleNamespace(
            total_input_tokens=100,
            total_output_tokens=25,
        ),
    )

    create_mock = AsyncMock(
        return_value=interaction,
    )

    monkeypatch.setattr(
        provider.client.aio.interactions,
        "create",
        create_mock,
    )

    response = await provider.generate(
        build_request()
    )

    assert response.content == (
        "Ethan Carter plays for the Yankees."
    )

    assert response.provider == "gemini"

    assert (
        response.model
        == "gemini-3.8-flash"
    )

    assert response.input_tokens == 100
    assert response.output_tokens == 25

    assert response.tool_calls == []

    # Incluso una respuesta sin tools conserva el ID de
    # interacción para permitir continuar con Gemini.
    assert response.provider_state is not None

    assert (
        response.provider_state.provider
        == "gemini"
    )

    assert (
        response.provider_state.data[
            "interaction_id"
        ]
        == "interaction-123"
    )

    create_mock.assert_awaited_once()

    kwargs = (
        create_mock.await_args.kwargs
    )

    assert kwargs["model"] == (
        "gemini-3.8-flash"
    )

    assert kwargs[
        "system_instruction"
    ] == (
        "You are Diamond IQ Analyst."
    )

    assert kwargs[
        "generation_config"
    ] == {
        "temperature": 0.2,
    }


@pytest.mark.anyio
async def test_gemini_provider_normalizes_function_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verifica que un function_call de Gemini sea normalizado
    a nuestro LLMToolCall.
    """

    provider = build_provider()

    tool_definition = {
        "type": "function",
        "function": {
            "name": "get_player",
            "description": (
                "Find players by name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {
                        "type": "string",
                    }
                },
                "required": [
                    "player_name",
                ],
            },
        },
    }

    function_call = SimpleNamespace(
        type="function_call",
        id="call-123",
        name="get_player",
        arguments={
            "player_name": (
                "Ethan Carter"
            ),
        },
    )

    interaction = SimpleNamespace(
        id="interaction-123",
        output_text="",
        steps=[
            function_call,
        ],
        usage=SimpleNamespace(
            total_input_tokens=120,
            total_output_tokens=15,
        ),
    )

    create_mock = AsyncMock(
        return_value=interaction,
    )

    monkeypatch.setattr(
        provider.client.aio.interactions,
        "create",
        create_mock,
    )

    response = await provider.generate(
        build_request(
            tools=[
                tool_definition,
            ]
        )
    )

    assert len(
        response.tool_calls
    ) == 1

    tool_call = response.tool_calls[0]

    assert tool_call.id == "call-123"

    assert (
        tool_call.name
        == "get_player"
    )

    assert tool_call.arguments == {
        "player_name": "Ethan Carter",
    }

    # Gemini debe devolver estado suficiente para poder
    # asociar posteriormente el resultado de la tool
    # con este function_call.
    assert response.provider_state is not None

    assert (
        response.provider_state.provider
        == "gemini"
    )

    assert (
        response.provider_state.data
        == {
            "interaction_id": (
                "interaction-123"
            ),
            "pending_tool_calls": {
                "call-123": (
                    "get_player"
                ),
            },
        }
    )

    kwargs = (
        create_mock.await_args.kwargs
    )

    assert kwargs["tools"] == [
        {
            "type": "function",
            "name": "get_player",
            "description": (
                "Find players by name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {
                        "type": (
                            "string"
                        ),
                    }
                },
                "required": [
                    "player_name",
                ],
            },
        }
    ]


@pytest.mark.anyio
async def test_gemini_provider_tracks_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verifica que Gemini reporte su consumo de tokens
    utilizando nuestro contrato común.
    """

    provider = build_provider()

    interaction = SimpleNamespace(
        id="interaction-usage",
        output_text="Hello.",
        steps=[],
        usage=SimpleNamespace(
            total_input_tokens=321,
            total_output_tokens=123,
        ),
    )

    create_mock = AsyncMock(
        return_value=interaction,
    )

    monkeypatch.setattr(
        provider.client.aio.interactions,
        "create",
        create_mock,
    )

    response = await provider.generate(
        build_request()
    )

    assert (
        response.input_tokens
        == 321
    )

    assert (
        response.output_tokens
        == 123
    )


@pytest.mark.anyio
async def test_gemini_provider_continues_interaction_with_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verifica que Gemini pueda continuar una interacción
    después de que Diamond IQ haya ejecutado una tool.

    El resultado debe enviarse como function_result utilizando
    el mismo call_id generado anteriormente por Gemini.
    """

    provider = build_provider()

    interaction = SimpleNamespace(
        id="interaction-2",
        output_text=(
            "Mateo Rivera has the highest Power."
        ),
        steps=[],
        usage=SimpleNamespace(
            total_input_tokens=150,
            total_output_tokens=40,
        ),
    )

    create_mock = AsyncMock(
        return_value=interaction,
    )

    monkeypatch.setattr(
        provider.client.aio.interactions,
        "create",
        create_mock,
    )

    request = LLMRequest(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Diamond IQ Analyst."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Who has the highest Power "
                    "on the Dodgers?"
                ),
            },
            {
                "role": "tool",
                "tool_call_id": (
                    "call-123"
                ),
                "name": (
                    "get_top_players"
                ),
                "content": (
                    '[{"name":"Mateo Rivera",'
                    '"value":98}]'
                ),
            },
        ],
        temperature=0.2,
        tools=[],
        provider_state=(
            LLMProviderState(
                provider="gemini",
                data={
                    "interaction_id": (
                        "interaction-1"
                    ),
                    "pending_tool_calls": {
                        "call-123": (
                            "get_top_players"
                        ),
                    },
                },
            )
        ),
    )

    await provider.generate(
        request,
    )

    kwargs = (
        create_mock.await_args.kwargs
    )

    assert (
        kwargs[
            "previous_interaction_id"
        ]
        == "interaction-1"
    )

    assert kwargs["input"] == [
        {
            "type": (
                "function_result"
            ),
            "name": (
                "get_top_players"
            ),
            "call_id": "call-123",
            "result": [
                {
                    "type": "text",
                    "text": (
                        '[{"name":"Mateo Rivera",'
                        '"value":98}]'
                    ),
                }
            ],
        }
    ]


def test_gemini_provider_maps_429_to_transient_error() -> None:
    """
    Un HTTP 429 debe considerarse un error temporal.

    Esto permite que LLMRouter aplique retry o fallback.
    """

    provider = build_provider()

    exc = FakeGeminiAPIError(
        code=429,
    )

    with pytest.raises(
        LLMProviderTransientError
    ):
        provider._raise_provider_error(
            exc
        )

def test_gemini_provider_maps_5xx_to_unavailable_error() -> None:
    """
    Un error 5xx representa indisponibilidad temporal
    del provider.
    """

    provider = build_provider()

    exc = FakeGeminiAPIError(
        code=503,
    )

    with pytest.raises(
        LLMProviderUnavailableError
    ):
        provider._raise_provider_error(
            exc
        )

def test_gemini_provider_maps_4xx_to_permanent_error() -> None:
    """
    Un error 4xx distinto de 429 normalmente representa
    un problema permanente de petición o configuración.
    """

    provider = build_provider()

    exc = FakeGeminiAPIError(
        code=400,
    )

    with pytest.raises(
        LLMProviderPermanentError
    ):
        provider._raise_provider_error(
            exc
        )