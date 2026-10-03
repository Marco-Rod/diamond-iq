from app.ai.tools.exceptions import ToolResultNotFoundError
from app.ai.tools.schemas import (
    ComparePlayersArguments,
    GetPlayerArguments,
    GetPlayerStatsArguments,
    GetTopPlayersArguments,
    PlayerComparisonResult,
    PlayerResult,
    PlayerStatsResult,
    TopPlayerResult,
)
from app.services.player import PlayerNotFoundError, PlayerService


async def get_top_players(
    *,
    arguments: GetTopPlayersArguments,
    player_service: PlayerService,
) -> list[TopPlayerResult]:
    """
    Ejecuta la tool `get_top_players`.

    Esta función actúa como adaptador entre:

    - argumentos estructurados generados por el LLM;
    - casos de uso ya existentes en PlayerService;
    - resultado compacto que devolveremos al modelo.

    La tool no contiene SQL y tampoco conoce PlayerRepository.
    """

    players = await player_service.get_top_players(
        metric=arguments.metric,
        team=arguments.team,
        limit=arguments.limit,
    )

    results: list[TopPlayerResult] = []

    for player in players:
        # `metric` ya está restringido por Literal, por lo que sabemos
        # que solo puede apuntar a uno de los ratings autorizados.
        metric_value = getattr(
            player,
            arguments.metric,
        )

        # El repository ya elimina ratings None, pero mantenemos esta
        # protección para no construir un resultado inválido si esa
        # implementación cambia en el futuro.
        if metric_value is None:
            continue

        results.append(
            TopPlayerResult(
                id=player.id,
                mlb_id=player.mlb_id,
                name=player.name,
                team=player.team,
                position=player.position,
                metric=arguments.metric,
                value=metric_value,
            )
        )

    return results

async def get_player(
    *,
    arguments: GetPlayerArguments,
    player_service: PlayerService,
) -> list[PlayerResult]:
    """
    Busca jugadores por nombre.

    Devolvemos una lista porque el nombre no es una clave única.
    Esto evita seleccionar silenciosamente al jugador equivocado cuando
    existen varias coincidencias.
    """

    players = await player_service.find_players_by_name(
        arguments.player_name,
    )
    if not players:
        raise ToolResultNotFoundError(
            f"No player was found with name '{arguments.player_name}'."
        )

    return [
        PlayerResult(
            id=player.id,
            mlb_id=player.mlb_id,
            name=player.name,
            team=player.team,
            position=player.position,
            contact=player.contact,
            power=player.power,
            vision=player.vision,
            overall=player.overall,
        )
        for player in players
    ]


async def get_player_stats(
    *,
    arguments: GetPlayerStatsArguments,
    player_service: PlayerService,
) -> list[PlayerStatsResult]:
    """
    Recupera estadísticas de temporada para jugadores que coincidan
    con el nombre solicitado.

    Un nombre inexistente se considera un error recuperable de tool,
    porque el LLM puede corregirlo utilizando contexto de otra llamada.
    """

    players = await player_service.find_players_by_name(
        arguments.player_name,
    )

    if not players:
        raise ToolResultNotFoundError(
            f"No player was found with name '{arguments.player_name}'."
        )

    results: list[PlayerStatsResult] = []

    for player in players:
        try:
            stats = await player_service.get_player_stats_by_season(
                player_id=player.id,
                season=arguments.season,
            )

        except PlayerNotFoundError:
            continue

        results.append(
            PlayerStatsResult(
                player_id=player.id,
                name=player.name,
                season=stats.season,
                games=stats.games,
                plate_appearances=stats.plate_appearances,
                hits=stats.hits,
                home_runs=stats.home_runs,
                rbi=stats.rbi,
                walks=stats.walks,
                strikeouts=stats.strikeouts,
                avg=stats.avg,
                obp=stats.obp,
                slg=stats.slg,
                ops=stats.ops,
                woba=stats.woba,
            )
        )
    if not results:
        raise ToolResultNotFoundError(
            f"No statistics were found for "
            f"'{arguments.player_name}' in season {arguments.season}."
        )
    
    return results

async def compare_players(
    *,
    arguments: ComparePlayersArguments,
    player_service: PlayerService,
) -> list[PlayerComparisonResult]:
    """
    Recupera los ratings solicitados para varios jugadores.

    El backend devuelve hechos estructurados; el LLM se encarga únicamente
    de redactar e interpretar las diferencias.
    """

    results: list[PlayerComparisonResult] = []

    for player_name in arguments.player_names:
        players = await player_service.find_players_by_name(
            player_name,
        )

        for player in players:
            ratings = {
                metric: getattr(player, metric)
                for metric in arguments.metrics
            }

            results.append(
                PlayerComparisonResult(
                    id=player.id,
                    name=player.name,
                    team=player.team,
                    position=player.position,
                    ratings=ratings,
                )
            )

    return results