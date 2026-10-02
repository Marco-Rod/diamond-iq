from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Player, PlayerSeasonStats


class PlayerRepository:
    """
    Encapsula el acceso a datos relacionado con jugadores.

    El repository concentra las consultas SQLAlchemy para evitar que
    endpoints, services o futuras tools del LLM dependan directamente
    de detalles de persistencia.

    Esto permite que las capas superiores trabajen con operaciones
    del dominio como `get_by_id()` o `get_by_name()` en lugar de construir
    consultas SQL directamente.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Recibe la sesión de base de datos mediante inyección de dependencia.

        El repository no crea su propia sesión porque la capa que inicia
        la operación debe controlar el ciclo de vida de la transacción.
        """
        self.session = session

    async def get_by_id(self, player_id: int) -> Player | None:
        """
        Recupera un jugador usando su identificador interno de Diamond IQ.

        Retorna None cuando el jugador no existe.
        """

        statement = select(Player).where(Player.id == player_id)

        result = await self.session.execute(statement)

        return result.scalar_one_or_none()

    async def get_by_mlb_id(self, mlb_id: int) -> Player | None:
        """
        Recupera un jugador mediante su identificador oficial de MLB.

        `mlb_id` tiene un índice único en PostgreSQL, por lo que esta
        consulta puede resolverse eficientemente incluso cuando aumente
        la cantidad de jugadores almacenados.
        """

        statement = select(Player).where(Player.mlb_id == mlb_id)

        result = await self.session.execute(statement)

        return result.scalar_one_or_none()

    async def get_by_name(
        self,
        name: str,
    ) -> list[Player]:
        """
        Recupera todos los jugadores cuyo nombre coincide exactamente.

        El nombre no es una identidad única: pueden existir varios jugadores
        con el mismo nombre. Por eso este método devuelve una lista en lugar
        de asumir que habrá un único resultado.

        Para identificar de manera inequívoca a un jugador debemos utilizar
        `id` o `mlb_id`.
        """

        statement = select(Player).where(Player.name == name).order_by(Player.id)

        result = await self.session.execute(statement)

        return list(result.scalars().all())

    async def list_players(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Player]:
        """
        Devuelve una lista paginada de jugadores.

        `limit` evita cargar toda la tabla en memoria accidentalmente.
        `offset` permite implementar paginación básica desde la API.
        """

        statement = select(Player).order_by(Player.name).limit(limit).offset(offset)

        result = await self.session.execute(statement)

        return list(result.scalars().all())

    async def get_with_stats(self, player_id: int) -> Player | None:
        """
        Recupera un jugador junto con sus estadísticas por temporada.

        `selectinload()` realiza una carga anticipada de la relación
        `season_stats`, evitando que el acceso posterior intente ejecutar
        consultas adicionales fuera del contexto async apropiado.
        """

        statement = (
            select(Player)
            .where(Player.id == player_id)
            .options(selectinload(Player.season_stats))
        )

        result = await self.session.execute(statement)

        return result.scalar_one_or_none()

    async def get_stats(
        self,
        player_id: int,
    ) -> list[PlayerSeasonStats]:
        """
        Recupera las estadísticas disponibles de un jugador.

        Se ordenan desde la temporada más reciente hacia la más antigua.
        """

        statement = (
            select(PlayerSeasonStats)
            .where(PlayerSeasonStats.player_id == player_id)
            .order_by(PlayerSeasonStats.season.desc())
        )

        result = await self.session.execute(statement)

        return list(result.scalars().all())

    async def add(self, player: Player) -> Player:
        """
        Añade un jugador a la sesión actual.

        `flush()` envía temporalmente los cambios a PostgreSQL para obtener
        valores generados por la base, como `player.id`, pero todavía no
        confirma definitivamente la transacción.

        El commit se deja para la capa de service, que debe decidir cuándo
        una operación de negocio completa puede considerarse exitosa.
        """

        self.session.add(player)

        await self.session.flush()

        return player

    async def get_top_players(
    self,
    *,
    metric: str,
    team: str | None = None,
    limit: int = 3,
    ) -> list[Player]:
        """
        Devuelve jugadores ordenados de mayor a menor según un rating.

        `metric` no se utiliza directamente como SQL arbitrario. Solo
        permitimos atributos conocidos del modelo para evitar consultas
        dinámicas inseguras.

        Esta función será especialmente útil más adelante para tools como
        `get_top_players`.
        """

        allowed_metrics = {
            "contact": Player.contact,
            "power": Player.power,
            "vision": Player.vision,
            "overall": Player.overall,
        }

        metric_column = allowed_metrics.get(metric)

        if metric_column is None:
            raise ValueError(f"Unsupported player metric: {metric}")

        statement = select(Player)

        if team is not None:
            statement = statement.where(
                Player.team == team,
            )

        statement = (
            statement.where(metric_column.is_not(None))
            .order_by(metric_column.desc())
            .limit(limit)
        )

        result = await self.session.execute(statement)

        return list(result.scalars().all())


    async def get_players_by_rating_profile(
        self,
        *,
        min_contact: int | None = None,
        max_power: int | None = None,
        limit: int = 20,
    ) -> list[Player]:
        """
        Busca jugadores que cumplan determinadas condiciones de ratings.

        Esta primera versión soporta el caso de demo:
        jugadores con Contact alto y Power bajo.

        Las condiciones se construyen de forma explícita en backend;
        no aceptamos SQL ni expresiones arbitrarias provenientes del usuario
        o del futuro LLM.
        """

        statement = select(Player)

        if min_contact is not None:
            statement = statement.where(
                Player.contact >= min_contact,
            )

        if max_power is not None:
            statement = statement.where(
                Player.power <= max_power,
            )

        statement = statement.order_by(Player.contact.desc()).limit(limit)

        result = await self.session.execute(statement)

        return list(result.scalars().all())