import pytest
from pydantic import ValidationError

from app.ai.tools.schemas import (
    ComparePlayersArguments,
    GetPlayerStatsArguments,
    GetTopPlayersArguments,
)


def test_get_top_players_arguments_accept_valid_values() -> None:
    arguments = GetTopPlayersArguments(
        metric="power",
        team="NYY",
        limit=3,
    )

    assert arguments.metric == "power"
    assert arguments.team == "NYY"
    assert arguments.limit == 3


def test_get_top_players_arguments_reject_unknown_metric() -> None:
    """
    El LLM no puede introducir arbitrariamente nombres de columnas
    o métricas fuera de la whitelist.
    """

    with pytest.raises(ValidationError):
        GetTopPlayersArguments(
            metric="salary",
            team="NYY",
        )


@pytest.mark.parametrize(
    "limit",
    [
        0,
        21,
    ],
)
def test_get_top_players_arguments_reject_invalid_limit(
    limit: int,
) -> None:
    with pytest.raises(ValidationError):
        GetTopPlayersArguments(
            metric="power",
            limit=limit,
        )

def test_compare_players_requires_at_least_two_players() -> None:
    with pytest.raises(ValidationError):
        ComparePlayersArguments(
            player_names=["Ethan Carter"],
        )

def test_get_player_stats_accepts_valid_arguments() -> None:
    arguments = GetPlayerStatsArguments(
        player_name="Ethan Carter",
        season=2026,
    )

    assert arguments.season == 2026