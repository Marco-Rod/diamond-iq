from unittest.mock import AsyncMock

import pytest

from app.ai.rag.schemas import (
    KnowledgeChunk,
    RetrievedChunk,
)
from app.ai.tools.exceptions import (
    ToolArgumentsError,
    ToolNotFoundError,
    ToolResultNotFoundError,
)
from app.ai.tools.registry import ToolRegistry


@pytest.fixture
def player_service() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def registry(
    player_service: AsyncMock,
) -> ToolRegistry:
    return ToolRegistry(
        player_service=player_service,
        knowledge_retriever=AsyncMock(),
    )

@pytest.mark.anyio
async def test_registry_rejects_unknown_tool(
    registry: ToolRegistry,
) -> None:
    with pytest.raises(ToolNotFoundError):
        await registry.execute(
            name="delete_database",
            arguments={},
        )

@pytest.mark.anyio
async def test_registry_rejects_invalid_arguments(
    registry: ToolRegistry,
) -> None:
    with pytest.raises(ToolArgumentsError):
        await registry.execute(
            name="get_top_players",
            arguments={
                "metric": "salary",
            },
        )

def test_registry_exposes_all_registered_tools() -> None:
    registry = ToolRegistry(
        player_service=AsyncMock(),
        knowledge_retriever=AsyncMock(),
    )

    definitions = registry.definitions()

    names = {
        item["function"]["name"]
        for item in definitions
    }

    assert names == {
        "get_player",
        "get_player_stats",
        "get_top_players",
        "compare_players",
        "search_knowledge",
    }

@pytest.mark.anyio
async def test_registry_propagates_result_not_found_error() -> None:
    player_service = AsyncMock()

    player_service.find_players_by_name.return_value = []

    registry = ToolRegistry(
        player_service=player_service,
        knowledge_retriever=AsyncMock(),
    )

    with pytest.raises(
        ToolResultNotFoundError,
        match="No player was found",
    ):
        await registry.execute(
            name="get_player_stats",
            arguments={
                "player_name": "X",
                "season": 2026,
            },
        )

@pytest.mark.anyio
async def test_tool_registry_executes_search_knowledge() -> None:
    """
    Verifica que el registry valide los argumentos y delegue
    la búsqueda al KnowledgeRetriever.
    """

    player_service = AsyncMock()
    knowledge_retriever = AsyncMock()

    knowledge_retriever.search.return_value = [
        RetrievedChunk(
            chunk=KnowledgeChunk(
                id="ratings.md:0",
                source="ratings.md",
                content=(
                    "Power represents a hitter's ability "
                    "to produce power when making contact."
                ),
            ),
            score=0.82,
        )
    ]

    registry = ToolRegistry(
        player_service=player_service,
        knowledge_retriever=(
            knowledge_retriever
        ),
    )

    result = await registry.execute(
    name="search_knowledge",
    arguments={
        "query": "What does Power mean?",
        "limit": 3,
    },
)

    assert len(result) == 1

    assert (
        result[0]["source"]
        == "ratings.md"
    )

    assert result[0]["score"] == 0.82

    knowledge_retriever.search.assert_awaited_once_with(
        query="What does Power mean?",
        limit=3,
    )

@pytest.mark.anyio
async def test_tool_registry_rejects_invalid_search_knowledge_arguments() -> None:
    """
    Verifica que search_knowledge respete la validación
    definida por Pydantic.
    """

    registry = ToolRegistry(
        player_service=AsyncMock(),
        knowledge_retriever=AsyncMock(),
    )

    with pytest.raises(
        ToolArgumentsError
    ):
        await registry.execute(
            name="search_knowledge",
            arguments={
                "query": "",
            },
        )

def test_tool_registry_exposes_search_knowledge_definition() -> None:
    """
    Verifica que el LLM pueda descubrir la tool de RAG.
    """

    registry = ToolRegistry(
        player_service=AsyncMock(),
        knowledge_retriever=AsyncMock(),
    )

    definitions = registry.definitions()

    names = {
        item["function"]["name"]
        for item in definitions
    }

    assert "search_knowledge" in names