from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.ai.limits import ToolCallLimitError
from app.ai.providers.exceptions import (
    LLMProviderUnavailableError,
)
from app.ai.schemas import (
    ChatResponse,
    ProviderMetadata,
    UsageInfo,
)
from app.api.v1.ai import get_ai_service
from app.main import app

client = TestClient(app)


def test_chat_returns_llm_response() -> None:
    """
    Verifica el contrato HTTP sin realizar una llamada real al LLM.

    Sustituimos AIService por un mock para que la suite permanezca
    rápida y determinista.
    """

    mock_service = AsyncMock()

    mock_service.chat.return_value = ChatResponse(
        answer="Ethan Carter has the highest Power.",
        provider=ProviderMetadata(
            provider="ollama",
            model="qwen2.5-coder:7b",
        ),
        usage=UsageInfo(
            input_tokens=50,
            output_tokens=12,
        ),
    )

    async def override_ai_service():
        return mock_service

    app.dependency_overrides[get_ai_service] = override_ai_service

    try:
        response = client.post(
            "/api/v1/ai/chat",
            json={
                "message": "Who has the highest Power?"
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert (
            body["answer"]
            == "Ethan Carter has the highest Power."
        )

        assert body["provider"] == {
            "provider": "ollama",
            "model": "qwen2.5-coder:7b",
        }

        assert body["usage"] == {
            "input_tokens": 50,
            "output_tokens": 12,
        }

        mock_service.chat.assert_awaited_once()

    finally:
        app.dependency_overrides.clear()


def test_chat_rejects_empty_message() -> None:
    response = client.post(
        "/api/v1/ai/chat",
        json={
            "message": ""
        },
    )

    assert response.status_code == 422

def test_chat_returns_429_when_execution_limit_is_exceeded() -> None:
    service = AsyncMock()

    service.chat.side_effect = ToolCallLimitError(
        "The conversation exceeded the maximum number of tool calls."
    )

    app.dependency_overrides[get_ai_service] = lambda: service

    client = TestClient(app)

    response = client.post(
        "/api/v1/ai/chat",
        json={
            "message": "Tell me about several players.",
        },
    )

    assert response.status_code == 429

    assert response.json() == {
        "detail": (
            "The conversation exceeded the maximum "
            "number of tool calls."
        )
    }

    app.dependency_overrides.clear()

def test_chat_returns_503_when_provider_is_unavailable() -> None:
    service = AsyncMock()

    service.chat.side_effect = (
        LLMProviderUnavailableError(
            "Could not connect to Ollama."
        )
    )

    app.dependency_overrides[
        get_ai_service
    ] = lambda: service

    client = TestClient(app)

    response = client.post(
        "/api/v1/ai/chat",
        json={
            "message": "Tell me about Ethan Carter.",
        },
    )

    assert response.status_code == 503

    assert response.json() == {
        "detail": (
            "The AI provider is temporarily unavailable."
        )
    }

    app.dependency_overrides.clear()