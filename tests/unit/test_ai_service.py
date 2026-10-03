from unittest.mock import AsyncMock, Mock

import pytest

from app.ai.guardrails import AIGuardrails
from app.ai.schemas import (
    ChatRequest,
    LLMResponse,
    LLMToolCall,
)
from app.ai.service import (
    MAX_TOOL_ITERATIONS,
    AIExecutionLimits,
    AIService,
    TokenBudgetExceededError,
    ToolArgumentsError,
    ToolCallLimitError,
    ToolIterationLimitError,
    ToolNotFoundError,
)
from app.ai.tools.exceptions import (
    ToolArgumentsError,
    ToolNotFoundError,
    ToolResultNotFoundError,
)


@pytest.mark.anyio
async def test_chat_returns_direct_response_without_tool_call() -> None:
    router = AsyncMock()

    tool_registry = Mock()
    tool_registry.definitions = Mock(
        return_value=[],
    )
    tool_registry.execute = AsyncMock()
    usage_tracker = AsyncMock()
    guardrails = Mock(spec=AIGuardrails)
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
        usage_tracker=usage_tracker,
        guardrails=guardrails
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
    usage_tracker = AsyncMock()
    guardrails = Mock(spec=AIGuardrails)
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
         usage_tracker=usage_tracker,
         guardrails=guardrails
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
    usage_tracker = AsyncMock()
    guardrails = Mock(spec=AIGuardrails)
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
        usage_tracker=usage_tracker,
        guardrails=guardrails
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
    usage_tracker = AsyncMock()
    guardrails = Mock(spec=AIGuardrails)
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
        usage_tracker=usage_tracker,
        guardrails=guardrails
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
    usage_tracker = AsyncMock()
    guardrails = Mock(spec=AIGuardrails)
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
        usage_tracker=usage_tracker,
        guardrails=guardrails
    )

    with pytest.raises(ToolIterationLimitError):
        await service.chat(
            ChatRequest(
                message="Keep searching forever."
            )
        )

    assert tool_registry.execute.await_count == MAX_TOOL_ITERATIONS


@pytest.mark.anyio
async def test_chat_can_recover_from_invalid_tool_arguments() -> None:
    router = AsyncMock()

    tool_registry = Mock()
    tool_registry.definitions = Mock(
        return_value=[]
    )
    tool_registry.execute = AsyncMock()

    invalid_call = LLMResponse(
        content=None,
        provider="ollama",
        model="qwen3:8b",
        input_tokens=100,
        output_tokens=20,
        tool_calls=[
            LLMToolCall(
                id=None,
                name="get_top_players",
                arguments={
                    "metric": "salary",
                    "team": "NYY",
                    "limit": 3,
                },
            )
        ],
    )

    corrected_call = LLMResponse(
        content=None,
        provider="ollama",
        model="qwen3:8b",
        input_tokens=120,
        output_tokens=20,
        tool_calls=[
            LLMToolCall(
                id=None,
                name="get_top_players",
                arguments={
                    "metric": "power",
                    "team": "NYY",
                    "limit": 3,
                },
            )
        ],
    )

    final_response = LLMResponse(
        content="The Yankees players with the highest Power are...",
        provider="ollama",
        model="qwen3:8b",
        input_tokens=150,
        output_tokens=30,
    )

    router.generate.side_effect = [
        invalid_call,
        corrected_call,
        final_response,
    ]
    usage_tracker = AsyncMock()
    guardrails = Mock(spec=AIGuardrails)
    tool_registry.execute.side_effect = [
        ToolArgumentsError(
            "Invalid metric 'salary'."
        ),
        [
            {
                "name": "Aaron Judge",
                "metric": "power",
                "value": 99,
            }
        ],
    ]

    service = AIService(
        router=router,
        tool_registry=tool_registry,
        usage_tracker=usage_tracker,
        guardrails=guardrails
    )

    response = await service.chat(
        ChatRequest(
            message="Who has the highest salary on the Yankees?"
        )
    )

    assert router.generate.await_count == 3
    assert tool_registry.execute.await_count == 2

    assert len(response.tools_used) == 2

    assert response.tools_used[0].success is False
    assert response.tools_used[0].error == "invalid_arguments"

    assert response.tools_used[1].success is True

@pytest.mark.anyio
async def test_chat_can_recover_from_unknown_tool() -> None:
    router = AsyncMock()

    tool_registry = Mock()
    tool_registry.definitions = Mock(
        return_value=[]
    )
    tool_registry.execute = AsyncMock()

    unknown_tool_response = LLMResponse(
        content=None,
        provider="ollama",
        model="qwen3:8b",
        input_tokens=50,
        output_tokens=10,
        tool_calls=[
            LLMToolCall(
                id=None,
                name="get_player_salary",
                arguments={
                    "player_name": "Ethan Carter",
                },
            )
        ],
    )

    final_response = LLMResponse(
        content=(
            "Diamond IQ does not provide salary information."
        ),
        provider="ollama",
        model="qwen3:8b",
        input_tokens=80,
        output_tokens=20,
    )

    router.generate.side_effect = [
        unknown_tool_response,
        final_response,
    ]
    usage_tracker = AsyncMock()
    guardrails = Mock(spec=AIGuardrails)
    tool_registry.execute.side_effect = ToolNotFoundError(
        "Unknown tool 'get_player_salary'."
    )

    service = AIService(
        router=router,
        tool_registry=tool_registry,
        usage_tracker=usage_tracker,
        guardrails=guardrails
    )

    response = await service.chat(
        ChatRequest(
            message="What is Ethan Carter's salary?"
        )
    )

    assert response.answer == (
        "Diamond IQ does not provide salary information."
    )

    assert response.tools_used[0].success is False
    assert response.tools_used[0].error == "tool_not_found"

@pytest.mark.anyio
async def test_chat_does_not_hide_unexpected_tool_errors() -> None:
    router = AsyncMock()

    tool_registry = Mock()
    tool_registry.definitions = Mock(
        return_value=[]
    )

    tool_registry.execute = AsyncMock(
        side_effect=RuntimeError(
            "Database connection lost."
        )
    )
    usage_tracker = AsyncMock()
    guardrails = Mock(spec=AIGuardrails)
    router.generate.return_value = LLMResponse(
        content=None,
        provider="ollama",
        model="qwen3:8b",
        input_tokens=50,
        output_tokens=10,
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

    service = AIService(
        router=router,
        tool_registry=tool_registry,
        usage_tracker=usage_tracker,
        guardrails=guardrails
    )

    with pytest.raises(
        RuntimeError,
        match="Database connection lost",
    ):
        await service.chat(
            ChatRequest(
                message="Tell me about Ethan Carter."
            )
        )

@pytest.mark.anyio
async def test_chat_stops_when_tool_call_limit_is_exceeded() -> None:
    router = AsyncMock()

    tool_registry = Mock()
    tool_registry.definitions = Mock(
        return_value=[]
    )
    tool_registry.execute = AsyncMock()
    usage_tracker = AsyncMock()
    guardrails = Mock(spec=AIGuardrails)
    router.generate.return_value = LLMResponse(
        content=None,
        provider="ollama",
        model="qwen3:8b",
        tool_calls=[
            LLMToolCall(
                name="get_player",
                arguments={"player_name": "Player A"},
            ),
            LLMToolCall(
                name="get_player",
                arguments={"player_name": "Player B"},
            ),
        ],
    )

    service = AIService(
        router=router,
        tool_registry=tool_registry,
        limits=AIExecutionLimits(
            max_tool_iterations=3,
            max_tool_calls_per_chat=1,
            max_total_tokens_per_chat=12_000,
        ),
        guardrails=guardrails,
        usage_tracker=usage_tracker,
    )

    with pytest.raises(ToolCallLimitError):
        await service.chat(
            ChatRequest(
                message="Tell me about these players."
            )
        )

    # El límite se comprueba antes de ejecutar ninguna tool.
    tool_registry.execute.assert_not_awaited()

@pytest.mark.anyio
async def test_chat_stops_when_token_budget_is_exhausted() -> None:
    router = AsyncMock()

    tool_registry = Mock()
    tool_registry.definitions = Mock(
        return_value=[]
    )
    tool_registry.execute = AsyncMock()
    usage_tracker = AsyncMock()
    guardrails = Mock(spec=AIGuardrails)
    router.generate.return_value = LLMResponse(
        content=None,
        provider="ollama",
        model="qwen3:8b",
        input_tokens=900,
        output_tokens=200,
        tool_calls=[
            LLMToolCall(
                name="get_player",
                arguments={
                    "player_name": "Ethan Carter",
                },
            )
        ],
    )

    service = AIService(
        router=router,
        tool_registry=tool_registry,
        limits=AIExecutionLimits(
            max_tool_iterations=3,
            max_tool_calls_per_chat=6,
            max_total_tokens_per_chat=1_000,
        ),
        guardrails=guardrails,
        usage_tracker=usage_tracker,
    )

    with pytest.raises(
        TokenBudgetExceededError
    ):
        await service.chat(
            ChatRequest(
                message="Tell me about Ethan Carter."
            )
        )

    tool_registry.execute.assert_not_awaited()

@pytest.mark.anyio
async def test_chat_can_recover_from_tool_result_not_found() -> None:
    router = AsyncMock()

    tool_registry = Mock()
    tool_registry.definitions = Mock(
        return_value=[]
    )
    tool_registry.execute = AsyncMock()

    usage_tracker = AsyncMock()

    first_response = LLMResponse(
        content=None,
        provider="ollama",
        model="qwen3:8b",
        input_tokens=100,
        output_tokens=20,
        tool_calls=[
            LLMToolCall(
                name="get_top_players",
                arguments={
                    "metric": "power",
                    "team": "LAD",
                    "limit": 1,
                },
            ),
            LLMToolCall(
                name="get_player_stats",
                arguments={
                    "player_name": "X",
                    "season": 2026,
                },
            ),
        ],
    )

    corrected_response = LLMResponse(
        content=None,
        provider="ollama",
        model="qwen3:8b",
        input_tokens=150,
        output_tokens=25,
        tool_calls=[
            LLMToolCall(
                name="get_player_stats",
                arguments={
                    "player_name": "Mateo Rivera",
                    "season": 2026,
                },
            )
        ],
    )

    final_response = LLMResponse(
        content="Mateo Rivera had 51 home runs in 2026.",
        provider="ollama",
        model="qwen3:8b",
        input_tokens=180,
        output_tokens=30,
    )

    router.generate.side_effect = [
        first_response,
        corrected_response,
        final_response,
    ]

    tool_registry.execute.side_effect = [
        # get_top_players
        [
            {
                "name": "Mateo Rivera",
                "metric": "power",
                "value": 98,
            }
        ],

        # get_player_stats("X")
        ToolResultNotFoundError(
            "No player was found with name 'X'."
        ),

        # corrected get_player_stats
        [
            {
                "name": "Mateo Rivera",
                "season": 2026,
                "home_runs": 51,
            }
        ],
    ]
    guardrails = Mock(spec=AIGuardrails)

    service = AIService(
        router=router,
        tool_registry=tool_registry,
        usage_tracker=usage_tracker,
        guardrails=guardrails,
    )

    response = await service.chat(
        ChatRequest(
            message=(
                "Who has the highest Power on the Dodgers, "
                "and how did that player perform in 2026?"
            )
        )
    )

    assert router.generate.await_count == 3
    assert tool_registry.execute.await_count == 3

    assert response.answer == (
        "Mateo Rivera had 51 home runs in 2026."
    )

    assert [
        execution.success
        for execution in response.tools_used
    ] == [
        True,
        False,
        True,
    ]

    assert response.tools_used[1].error == "result_not_found"


@pytest.mark.anyio
async def test_chat_does_not_validate_output_before_tool_execution() -> None:
    """
    Verifica que una respuesta intermedia que contiene tool calls
    no sea tratada como una respuesta final para el usuario.

    Un provider puede devolver una tool call sin contenido textual.
    En ese caso, AIService debe ejecutar primero la tool y continuar
    la conversación con el LLM.

    El guardrail de salida solo debe validar la respuesta final,
    cuando ya no existen tool calls pendientes.
    """

    # El router es asíncrono porque generate() realiza la llamada
    # al provider de LLM.
    router = AsyncMock()

    # ToolRegistry mezcla métodos síncronos y asíncronos:
    #
    # - definitions() es síncrono y devuelve las tools disponibles.
    # - execute() es asíncrono porque algunas tools acceden a BD,
    #   embeddings u otros recursos async.
    tool_registry = Mock()
    tool_registry.execute = AsyncMock()

    # Simulamos el resultado real que devolvería search_knowledge
    # después de recuperar información desde la base de conocimiento.
    tool_registry.execute.return_value = [
        {
            "source": "ratings.md",
            "content": (
                "Power represents a hitter's ability "
                "to produce power when making contact."
            ),
            "score": 0.82,
        }
    ]

    # definitions() debe devolver una lista normal, no una coroutine.
    tool_registry.definitions.return_value = [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": (
                    "Search Diamond IQ knowledge."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                        },
                        "limit": {
                            "type": "integer",
                        },
                    },
                    "required": [
                        "query",
                    ],
                },
            },
        }
    ]

    usage_tracker = AsyncMock()

    # Usamos un Mock normal porque los métodos de AIGuardrails
    # son validaciones síncronas.
    guardrails = Mock(
        spec=AIGuardrails
    )

    # Primera respuesta del modelo:
    #
    # El modelo todavía no responde al usuario.
    # Solicita ejecutar search_knowledge y por ello puede devolver
    # content vacío.
    #
    # Segunda respuesta:
    #
    # Después de recibir el resultado de la tool, el modelo genera
    # la respuesta textual final.
    router.generate.side_effect = [
        LLMResponse(
            content="",
            provider="gemini",
            model="gemini-3.8-flash",
            tool_calls=[
                LLMToolCall(
                    id="call-1",
                    name="search_knowledge",
                    arguments={
                        "query": (
                            "What does Power mean "
                            "in Diamond IQ?"
                        ),
                        "limit": 3,
                    },
                )
            ],
        ),
        LLMResponse(
            content=(
                "Power represents a hitter's ability "
                "to produce power when making contact."
            ),
            provider="gemini",
            model="gemini-3.8-flash",
        ),
    ]

    service = AIService(
        router=router,
        tool_registry=tool_registry,
        usage_tracker=usage_tracker,
        guardrails=guardrails,
        limits=AIExecutionLimits(),
    )

    await service.chat(
        ChatRequest(
            message=(
                "What does Power mean "
                "in Diamond IQ?"
            )
        )
    )

    # La tool debe haberse ejecutado utilizando exactamente
    # los argumentos solicitados por el modelo.
    tool_registry.execute.assert_awaited_once_with(
        name="search_knowledge",
        arguments={
            "query": (
                "What does Power mean "
                "in Diamond IQ?"
            ),
            "limit": 3,
        },
    )

    # El output vacío de la primera respuesta NO debe validarse.
    #
    # validate_output() debe ejecutarse una sola vez y únicamente
    # sobre la respuesta final destinada al usuario.
    guardrails.validate_output.assert_called_once_with(
        "Power represents a hitter's ability "
        "to produce power when making contact."
    )