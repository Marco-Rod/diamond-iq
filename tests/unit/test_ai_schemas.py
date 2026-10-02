import pytest
from pydantic import ValidationError

from app.ai.schemas import (
    ChatRequest,
    ChatResponse,
    ProviderMetadata,
    ToolExecution,
    UsageInfo,
)


def test_chat_request_accepts_valid_message() -> None:
    request = ChatRequest(
        message="¿Quién tiene mayor Power en Yankees?"
    )

    assert request.message == "¿Quién tiene mayor Power en Yankees?"


def test_chat_request_rejects_empty_message() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_usage_info_defaults_to_zero() -> None:
    usage = UsageInfo()

    assert usage.input_tokens == 0
    assert usage.output_tokens == 0


def test_usage_info_rejects_negative_tokens() -> None:
    with pytest.raises(ValidationError):
        UsageInfo(
            input_tokens=-1,
            output_tokens=10,
        )


def test_chat_response_supports_provider_and_tools() -> None:
    response = ChatResponse(
        answer="Ethan Carter tiene el Power más alto.",
        provider=ProviderMetadata(
            provider="ollama",
            model="qwen3:4b",
        ),
        tools_used=[
            ToolExecution(
                name="get_top_players",
                success=True,
            )
        ],
        usage=UsageInfo(
            input_tokens=100,
            output_tokens=25,
        ),
    )

    assert response.provider.provider == "ollama"
    assert response.provider.model == "qwen3:4b"
    assert response.tools_used[0].name == "get_top_players"
    assert response.usage.input_tokens == 100