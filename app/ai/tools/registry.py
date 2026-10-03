from typing import Any

from pydantic import ValidationError

from app.ai.tools.exceptions import (
    ToolArgumentsError,
    ToolNotFoundError,
    ToolResultNotFoundError,
)
from app.ai.tools.players import (
    compare_players,
    get_player,
    get_player_stats,
    get_top_players,
)
from app.ai.tools.schemas import (
    ComparePlayersArguments,
    GetPlayerArguments,
    GetPlayerStatsArguments,
    GetTopPlayersArguments,
)
from app.services.player import PlayerService

GET_TOP_PLAYERS_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_top_players",
        "description": (
            "Returns Diamond IQ players ranked by a rating such as "
            "contact, power, vision or overall. "
            "Use this tool whenever the user asks which players have "
            "the highest rating or requests top players by rating."
        ),
        "parameters": GetTopPlayersArguments.model_json_schema(),
    },
}

GET_PLAYER_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_player",
        "description": (
            "Finds Diamond IQ players by name. "
            "Use this tool when the user asks about a specific player."
        ),
        "parameters": GetPlayerArguments.model_json_schema(),
    },
}

GET_PLAYER_STATS_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_player_stats",
        "description": (
            "Returns offensive season statistics for a Diamond IQ player."
        ),
        "parameters": GetPlayerStatsArguments.model_json_schema(),
    },
}

COMPARE_PLAYERS_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "compare_players",
        "description": (
            "Returns structured Diamond IQ ratings for multiple players "
            "so their profiles can be compared."
        ),
        "parameters": ComparePlayersArguments.model_json_schema(),
    },
}


class ToolRegistry:
    """
    Registro central de tools disponibles para el LLM.

    Tiene dos responsabilidades:

    1. exponer las definiciones que enviamos al modelo;
    2. ejecutar únicamente tools conocidas y validadas.

    El LLM nunca recibe acceso directo a Python, SQLAlchemy o PostgreSQL.
    """

    def __init__(
        self,
        *,
        player_service: PlayerService,
    ) -> None:
        self.player_service = player_service

    def definitions(self) -> list[dict[str, Any]]:
        """
        Devuelve las tools que podrán ser incluidas dentro de LLMRequest.
        """

        return [
            GET_PLAYER_DEFINITION,
            GET_PLAYER_STATS_DEFINITION,
            GET_TOP_PLAYERS_DEFINITION,
            COMPARE_PLAYERS_DEFINITION,
        ]

    async def execute(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Ejecuta una tool registrada.

        La ejecución es explícita: no utilizamos eval(), imports dinámicos
        ni nombres arbitrarios provenientes del modelo.
        """

        if name == "get_top_players":
            try:
                parsed_arguments = GetTopPlayersArguments.model_validate(
                    arguments,
                )
            except ValidationError as exc:
                raise ToolArgumentsError(
                    f"Invalid arguments for tool '{name}': "
                    f"{exc.errors(include_url=False)}"
                ) from exc

            result = await get_top_players(
                arguments=parsed_arguments,
                player_service=self.player_service,
            )

            # Convertimos los modelos Pydantic a estructuras JSON-safe.
            return [
                item.model_dump(
                    mode="json",
                    by_alias=True,
                )
                for item in result
            ]

        if name == "get_player":
            try:
                parsed_arguments = GetPlayerArguments.model_validate(
                    arguments,
                )
            except ValidationError as exc:
                raise ToolArgumentsError(
                    f"Invalid arguments for tool '{name}'."
                ) from exc

            result = await get_player(
                arguments=parsed_arguments,
                player_service=self.player_service,
            )

            return [
                item.model_dump(
                    mode="json",
                    by_alias=True,
                )
                for item in result
            ]

        if name == "get_player_stats":
            try:
                parsed_arguments = GetPlayerStatsArguments.model_validate(
                    arguments,
                )
            except ValidationError as exc:
                raise ToolArgumentsError(
                    f"Invalid arguments for tool '{name}'."
                ) from exc

            result = await get_player_stats(
                arguments=parsed_arguments,
                player_service=self.player_service,
            )

            return [
                item.model_dump(
                    mode="json",
                    by_alias=True,
                )
                for item in result
            ]

        if name == "compare_players":
            try:
                parsed_arguments = ComparePlayersArguments.model_validate(
                    arguments,
                )
            except ValidationError as exc:
                raise ToolArgumentsError(
                    f"Invalid arguments for tool '{name}'."
                ) from exc

            result = await compare_players(
                arguments=parsed_arguments,
                player_service=self.player_service,
            )

            return [
                item.model_dump(
                    mode="json",
                    by_alias=True,
                )
                for item in result
            ]
        
        raise ToolNotFoundError(
            f"Tool '{name}' is not registered."
        )