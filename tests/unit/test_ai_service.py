from unittest.mock import AsyncMock, Mock

import pytest

from app.ai.schemas import (
    ChatRequest,
    LLMResponse,
    LLMToolCall,
)
from app.ai.service import MAX_TOOL_ITERATIONS, AIService, ToolIterationLimitError


@pytest.mark.anyio
async def test_chat_returns_direct_response_without_tool_call() -> None:
    router = AsyncMock()

    tool_registry = Mock()
    tool_registry.definitions = Mock(
        return_value=[],
    )
    tool_registry.execute = AsyncMock()

    router.generate.return_value = LLMResponse(
        content="Baseball is a bat-and-ball sport.",
        provider="ollama",
        model="qwen3:8b",
        input_tokens=20,
        output_tokens=10,
        finish_reason="stop",
    )

    service = AIService(
        router=router,
        tool_registry=tool_registry,
    )

    response = await service.chat(
        ChatRequest(
            message="What is baseball?"
        )
    )

    assert response.answer == "Baseball is a bat-and-ball sport."
    assert response.tools_used == []

    tool_registry.execute.assert_not_awaited()

@pytest.mark.anyio
async def test_chat_executes_tool_and_requests_final_response() -> None:
    router = AsyncMock()
    tool_registry = Mock()

    tool_registry.definitions = Mock(
        return_value=[
            {
                "type": "function",
                "function": {
                    "name": "get_top_players",
                },
            }
        ]
    )

    tool_registry.execute = AsyncMock(
        return_value=[
            {
                "id": 1,
                "mlb_id": 900001,
                "name": "Ethan Carter",
                "team": "NYY",
                "position": "RF",
                "metric": "power",
                "value": 96,
            }
        ]
    )

    first_response = LLMResponse(
        content=None,
        provider="ollama",
        model="qwen2.5-coder:7b",
        input_tokens=40,
        output_tokens=10,
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

    final_response = LLMResponse(
        content=(
            "Ethan Carter leads the Yankees with a Power rating of 96."
        ),
        provider="ollama",
        model="qwen2.5-coder:7b",
        input_tokens=80,
        output_tokens=20,
        finish_reason="stop",
    )

    router.generate.side_effect = [
        first_response,
        final_response,
    ]

    tool_registry.execute.return_value = [
        {
            "id": 1,
            "mlb_id": 900001,
            "name": "Ethan Carter",
            "team": "NYY",
            "position": "RF",
            "metric": "power",
            "value": 96,
        }
    ]

    service = AIService(
        router=router,
        tool_registry=tool_registry,
    )

    response = await service.chat(
        ChatRequest(
            message="Who has the highest Power on the Yankees?"
        )
    )

    assert (
        response.answer
        == "Ethan Carter leads the Yankees with a Power rating of 96."
    )

    assert len(response.tools_used) == 1

    assert (
        response.tools_used[0].name
        == "get_top_players"
    )

    assert response.usage.input_tokens == 120
    assert response.usage.output_tokens == 30

    tool_registry.execute.assert_awaited_once_with(
        name="get_top_players",
        arguments={
            "team": "NYY",
            "metric": "power",
            "limit": 3,
        },
    )

    assert router.generate.await_count == 2


@pytest.mark.anyio
async def test_chat_executes_multiple_tool_calls_in_same_iteration() -> None:
    router = AsyncMock()

    tool_registry = Mock()
    tool_registry.definitions = Mock(
        return_value=[
            {
                "type": "function",
                "function": {
                    "name": "get_player",
                },
            }
        ]
    )
    tool_registry.execute = AsyncMock()

    first_response = LLMResponse(
        content=None,
        provider="ollama",
        model="qwen3:8b",
        input_tokens=100,
        output_tokens=20,
        tool_calls=[
            LLMToolCall(
                id=None,
                name="get_player",
                arguments={
                    "player_name": "Ethan Carter",
                },
            ),
            LLMToolCall(
                id=None,
                name="get_player",
                arguments={
                    "player_name": "Mateo Rivera",
                },
            ),
        ],
    )

    final_response = LLMResponse(
        content="Ethan Carter and Mateo Rivera were found.",
        provider="ollama",
        model="qwen3:8b",
        input_tokens=150,
        output_tokens=30,
        finish_reason="stop",
    )

    router.generate.side_effect = [
        first_response,
        final_response,
    ]

    tool_registry.execute.side_effect = [
        [
            {
                "name": "Ethan Carter",
            }
        ],
        [
            {
                "name": "Mateo Rivera",
            }
        ],
    ]

    service = AIService(
        router=router,
        tool_registry=tool_registry,
    )

    response = await service.chat(
        ChatRequest(
            message="Tell me about Ethan Carter and Mateo Rivera."
        )
    )

    assert router.generate.await_count == 2

    assert tool_registry.execute.await_count == 2

    assert len(response.tools_used) == 2

    assert response.usage.input_tokens == 250
    assert response.usage.output_tokens == 50


@pytest.mark.anyio
async def test_chat_can_request_another_tool_in_next_iteration() -> None:
    router = AsyncMock()

    tool_registry = Mock()
    tool_registry.definitions = Mock(
        return_value=[]
    )
    tool_registry.execute = AsyncMock()

    first_response = LLMResponse(
        content=None,
        provider="ollama",
        model="qwen3:8b",
        input_tokens=100,
        output_tokens=20,
        tool_calls=[
            LLMToolCall(
                id=None,
                name="get_player",
                arguments={
                    "player_name": "Ethan Carter",
                },
            )
        ],
    )

    second_response = LLMResponse(
        content=None,
        provider="ollama",
        model="qwen3:8b",
        input_tokens=120,
        output_tokens=20,
        tool_calls=[
            LLMToolCall(
                id=None,
                name="get_player_stats",
                arguments={
                    "player_name": "Ethan Carter",
                    "season": 2026,
                },
            )
        ],
    )

    final_response = LLMResponse(
        content="Ethan Carter had a strong 2026 season.",
        provider="ollama",
        model="qwen3:8b",
        input_tokens=150,
        output_tokens=40,
    )

    router.generate.side_effect = [
        first_response,
        second_response,
        final_response,
    ]

    tool_registry.execute.side_effect = [
        [{"name": "Ethan Carter"}],
        [
            {
                "name": "Ethan Carter",
                "season": 2026,
            }
        ],
    ]

    service = AIService(
        router=router,
        tool_registry=tool_registry,
    )

    response = await service.chat(
        ChatRequest(
            message="Tell me about Ethan Carter's 2026 season."
        )
    )

    assert router.generate.await_count == 3
    assert tool_registry.execute.await_count == 2

    assert [
        execution.name
        for execution in response.tools_used
    ] == [
        "get_player",
        "get_player_stats",
    ]


@pytest.mark.anyio
async def test_chat_stops_after_maximum_tool_iterations() -> None:
    router = AsyncMock()

    tool_registry = Mock()
    tool_registry.definitions = Mock(
        return_value=[]
    )
    tool_registry.execute = AsyncMock(
        return_value=[]
    )

    endless_tool_response = LLMResponse(
        content=None,
        provider="ollama",
        model="qwen3:8b",
        input_tokens=10,
        output_tokens=5,
        tool_calls=[
            LLMToolCall(
                id=None,
                name="get_player",
                arguments={
                    "player_name": "Ethan Carter",
                },
            )
        ],
    )

    router.generate.return_value = endless_tool_response

    service = AIService(
        router=router,
        tool_registry=tool_registry,
    )

    with pytest.raises(ToolIterationLimitError):
        await service.chat(
            ChatRequest(
                message="Keep searching forever."
            )
        )

    assert tool_registry.execute.await_count == MAX_TOOL_ITERATIONS