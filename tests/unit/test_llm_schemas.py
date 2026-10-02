from app.ai.schemas import (
    LLMRequest,
    LLMResponse,
    LLMToolCall,
)


def test_llm_request_uses_default_temperature() -> None:
    request = LLMRequest(
        messages=[
            {
                "role": "user",
                "content": "Who has the highest Power?"
            }
        ]
    )

    assert request.temperature == 0.2
    assert request.tools == []


def test_llm_response_supports_text_response() -> None:
    response = LLMResponse(
        content="Ethan Carter has the highest Power.",
        provider="ollama",
        model="qwen3:4b",
        input_tokens=50,
        output_tokens=12,
        finish_reason="stop",
    )

    assert response.content is not None
    assert response.tool_calls == []


def test_llm_response_supports_tool_call() -> None:
    response = LLMResponse(
        provider="ollama",
        model="qwen3:4b",
        tool_calls=[
            LLMToolCall(
                id="tool-1",
                name="get_top_players",
                arguments={
                    "team": "NYY",
                    "metric": "power",
                    "limit": 3,
                },
            )
        ],
    )

    assert response.content is None

    assert len(response.tool_calls) == 1

    tool_call = response.tool_calls[0]

    assert tool_call.name == "get_top_players"
    assert tool_call.arguments["metric"] == "power"