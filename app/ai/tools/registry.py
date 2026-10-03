from typing import Any

from pydantic import ValidationError

from app.ai.rag.retriever import KnowledgeRetriever
from app.ai.tools.exceptions import (
    ToolArgumentsError,
    ToolNotFoundError,
    ToolResultNotFoundError,
)
from app.ai.tools.knowledge import (
    search_knowledge,
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
    SearchKnowledgeArguments,
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
            "Finds one or more Diamond IQ players by name and returns "
            "their stored ratings. Use this tool when the user asks for "
            "the ratings or profile of a specific player."
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
            "Compares Diamond IQ ratings for two or more players. "
            "Use this tool only when the user explicitly asks to compare "
            "multiple players. Do not use it to retrieve the rating of "
            "a single player; use get_player instead."
        ),
        "parameters": ComparePlayersArguments.model_json_schema(),
    },
}

SEARCH_KNOWLEDGE: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": (
            "Busca información en la documentación y metodología "
            "de Diamond IQ. Úsala para preguntas conceptuales, "
            "definiciones de ratings, metodología y contenido "
            "explicativo. No debe utilizarse para consultar "
            "ratings o estadísticas dinámicas de jugadores."
        ),
        "parameters": (
            SearchKnowledgeArguments.model_json_schema()
        ),
    },
}

class ToolRegistry:
    """
    Registro central de tools disponibles para el LLM.

    Tiene dos responsabilidades:

    1. exponer las definiciones que enviamos al modelo;
    2. ejecutar únicamente tools conocidas y validadas.
    
    El registro actúa como whitelist:
    El LLM nunca recibe acceso directo a Python, SQLAlchemy o PostgreSQL.
    """

    def __init__(
        self,
        *,
        player_service: PlayerService,
        knowledge_retriever: KnowledgeRetriever,
    ) -> None:
        """
        Inicializa el registro de tools disponibles para el LLM.

        El registro funciona como una whitelist explícita:
        el modelo solo puede ejecutar capacidades registradas aquí.
        """
        self.player_service = player_service
        self.knowledge_retriever = (
            knowledge_retriever
        )

    def definitions(self) -> list[dict[str, Any]]:
        """
        Devuelve las tools que podrán ser incluidas dentro de LLMRequest.
        """
        
        return [
            GET_PLAYER_DEFINITION,
            GET_PLAYER_STATS_DEFINITION,
            GET_TOP_PLAYERS_DEFINITION,
            COMPARE_PLAYERS_DEFINITION,
            SEARCH_KNOWLEDGE,
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

        if name == "search_knowledge":
            try:
                # Validamos los argumentos generados por el LLM antes
                # de enviarlos al componente de recuperación.
                parsed_arguments = (
                    SearchKnowledgeArguments.model_validate(
                        arguments
                    )
                )

            except ValidationError as exc:
                raise ToolArgumentsError(
                    f"Invalid arguments for tool '{name}'."
                ) from exc

            results = await search_knowledge(
                arguments=parsed_arguments,
                retriever=self.knowledge_retriever,
            )

            return [
                result.model_dump(
                    mode="json"
                )
                for result in results
            ]
        raise ToolNotFoundError(
            f"Tool '{name}' is not registered."
        )