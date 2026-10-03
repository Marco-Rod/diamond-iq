from unittest.mock import AsyncMock

import pytest

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

def test_registry_exposes_all_player_tools(
    registry: ToolRegistry,
) -> None:
    definitions = registry.definitions()

    names = {
        definition["function"]["name"]
        for definition in definitions
    }

    assert names == {
        "get_player",
        "get_player_stats",
        "get_top_players",
        "compare_players",
    }

@pytest.mark.anyio
async def test_registry_propagates_result_not_found_error() -> None:
    player_service = AsyncMock()

    player_service.find_players_by_name.return_value = []

    registry = ToolRegistry(
        player_service=player_service,
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