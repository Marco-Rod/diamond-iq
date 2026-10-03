from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Player, PlayerSeasonStats
from app.repositories.player import PlayerRepository
from app.schemas.player import PlayerCreate


class UnsupportedPlayerMetricError(Exception):
    """
    Se lanza cuando una consulta intenta utilizar un rating
    que Diamond IQ no reconoce como métrica válida.
    """


class PlayerAlreadyExistsError(Exception):
    """
    Se lanza cuando intentamos crear un jugador cuyo `mlb_id`
    ya existe en Diamond IQ.

    La excepción pertenece a la capa de negocio, no a FastAPI.
    Más adelante la API decidirá cómo convertirla a una respuesta HTTP.
    """


class PlayerNotFoundError(Exception):
    """
    Se lanza cuando una operación requiere un jugador existente
    y no podemos encontrarlo.
    """


class PlayerService:
    """
    Coordina los casos de uso relacionados con jugadores.

    El service contiene reglas de negocio y orquesta repositories,
    transacciones y validaciones que van más allá de la validación
    estructural realizada por Pydantic.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        El service recibe la sesión y construye el repository que necesita.

        De esta manera todas las operaciones realizadas durante el caso
        de uso comparten la misma sesión y la misma transacción.
        """
        self.session = session
        self.repository = PlayerRepository(session)

    async def create_player(
        self,
        data: PlayerCreate,
    ) -> Player:
        """
        Crea un jugador dentro de Diamond IQ.

        Flujo:
        1. comprueba si `mlb_id` ya existe;
        2. construye el modelo ORM;
        3. delega la persistencia al repository;
        4. confirma la transacción;
        5. devuelve el jugador persistido.
        """

        existing_player = await self.repository.get_by_mlb_id(
            data.mlb_id,
        )

        if existing_player is not None:
            raise PlayerAlreadyExistsError(
                f"Player with MLB id {data.mlb_id} already exists."
            )

        player = Player(
            mlb_id=data.mlb_id,
            name=data.name,
            team=data.team,
            position=data.position,
            contact=data.contact,
            power=data.power,
            vision=data.vision,
            overall=data.overall,
        )

        try:
            await self.repository.add(player)

            # `commit()` confirma definitivamente la transacción.
            #
            # El repository solamente hizo `flush()`, así que hasta este
            # momento todavía podíamos hacer rollback de toda la operación.
            await self.session.commit()

        except IntegrityError:
            # Una restricción de PostgreSQL todavía podría fallar aunque
            # hayamos realizado validaciones previas.
            #
            # Ejemplo: dos requests concurrentes intentan crear el mismo
            # `mlb_id` casi al mismo tiempo.
            await self.session.rollback()

            raise PlayerAlreadyExistsError(
                f"Player with MLB id {data.mlb_id} already exists."
            ) from None

        # `refresh()` vuelve a leer desde PostgreSQL los valores del modelo.
        # Es útil para asegurarnos de tener campos generados por la DB,
        # como el identificador primario.
        await self.session.refresh(player)

        return player

    async def list_players(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Player]:
        """
        Devuelve jugadores utilizando paginación básica.

        Las reglas de límites máximos pueden endurecerse posteriormente
        para impedir requests demasiado grandes.
        """

        return await self.repository.list_players(
            limit=limit,
            offset=offset,
        )

    async def get_player(
        self,
        player_id: int,
    ) -> Player:
        """
        Recupera un jugador por su identificador interno.

        A diferencia del repository, este método no devuelve None:
        desde el punto de vista del caso de uso, no encontrar al jugador
        constituye un error de negocio.
        """

        player = await self.repository.get_by_id(player_id)

        if player is None:
            raise PlayerNotFoundError(f"Player {player_id} was not found.")

        return player

    async def get_player_with_stats(
        self,
        player_id: int,
    ) -> Player:
        """
        Recupera un jugador incluyendo sus estadísticas por temporada.
        """

        player = await self.repository.get_with_stats(player_id)

        if player is None:
            raise PlayerNotFoundError(f"Player {player_id} was not found.")

        return player

    async def find_players_by_name(
        self,
        name: str,
    ) -> list[Player]:
        """
        Busca jugadores por nombre exacto.

        Devuelve una lista porque el nombre no constituye una identidad
        única y pueden existir varios jugadores con el mismo nombre.
        """

        return await self.repository.get_by_name(name)


    async def get_top_players(
        self,
        *,
        metric: str,
        team: str | None = None,
        limit: int = 3,
    ) -> list[Player]:
        """
        Recupera los jugadores mejor evaluados según una métrica.

        El service valida parámetros de negocio antes de delegar
        la consulta al repository.
        """

        allowed_metrics = {
            "contact",
            "power",
            "vision",
            "overall",
        }

        if metric not in allowed_metrics:
            raise UnsupportedPlayerMetricError(f"Unsupported player metric: {metric}")

        if limit < 1 or limit > 20:
            raise ValueError("limit must be between 1 and 20")

        return await self.repository.get_top_players(
            metric=metric,
            team=team,
            limit=limit,
        )


    async def find_players_by_rating_profile(
        self,
        *,
        min_contact: int | None = None,
        max_power: int | None = None,
        limit: int = 20,
    ) -> list[Player]:
        """
        Recupera jugadores que coincidan con un perfil de ratings.

        Esta operación representa una consulta de dominio, no una consulta
        SQL libre. Las futuras tools del LLM podrán reutilizarla de manera
        segura.
        """

        return await self.repository.get_players_by_rating_profile(
            min_contact=min_contact,
            max_power=max_power,
            limit=limit,
        )

    async def get_player_stats_by_season(
    self,
    *,
    player_id: int,
    season: int,
    ) -> PlayerSeasonStats:
        """
        Recupera estadísticas de temporada y convierte ausencia de datos
        en un error de dominio.
        """

        stats = await self.repository.get_stats_by_season(
            player_id=player_id,
            season=season,
        )

        if stats is None:
            raise PlayerNotFoundError(
                f"Stats for player {player_id} in season {season} were not found."
            )

        return stats
