from decimal import Decimal

from sqlalchemy import (
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Player(Base):
    """
    Representa la identidad principal de un jugador dentro de Diamond IQ.

    Esta tabla contiene información relativamente estable del jugador,
    como su identificador MLB, equipo, posición y ratings actuales.

    Las estadísticas que cambian por temporada viven en
    `PlayerSeasonStats`, evitando duplicar la identidad del jugador
    para cada temporada.
    """

    __tablename__ = "players"

    # Identificador interno de Diamond IQ.
    #
    # No dependemos de MLB para nuestras relaciones internas porque
    # queremos mantener control sobre la identidad dentro de nuestra DB.
    id: Mapped[int] = mapped_column(primary_key=True)

    # Identificador oficial del jugador en MLB.
    #
    # Es más seguro que utilizar `name` como identificador externo,
    # ya que pueden existir jugadores con nombres iguales o similares.
    mlb_id: Mapped[int] = mapped_column(
        Integer,
        unique=True,
        nullable=False,
        index=True,
    )

    # Indexamos el nombre porque será una de las formas más frecuentes
    # de localizar jugadores desde la API y, posteriormente, desde tools
    # utilizadas por el LLM.
    name: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
        index=True,
    )

    team: Mapped[str | None] = mapped_column(String(80))
    position: Mapped[str | None] = mapped_column(String(10))

    # Ratings de Diamond IQ.
    #
    # Se mantienen opcionales porque podemos almacenar jugadores que
    # todavía no hayan pasado por el pipeline de cálculo de ratings.
    contact: Mapped[int | None] = mapped_column(Integer)
    power: Mapped[int | None] = mapped_column(Integer)
    vision: Mapped[int | None] = mapped_column(Integer)
    overall: Mapped[int | None] = mapped_column(Integer)

    # Relación one-to-many:
    #
    # Un jugador puede tener estadísticas para múltiples temporadas.
    #
    # `delete-orphan` evita conservar estadísticas sin un Player padre,
    # mientras que `all` propaga operaciones ORM relevantes hacia ellas.
    season_stats: Mapped[list["PlayerSeasonStats"]] = relationship(
        back_populates="player",
        cascade="all, delete-orphan",
    )


class PlayerSeasonStats(Base):
    """
    Contiene las estadísticas de un jugador para una temporada específica.

    La combinación `(player_id, season)` es única para impedir que
    Diamond IQ almacene dos registros distintos para el mismo jugador
    y la misma temporada.
    """

    __tablename__ = "player_season_stats"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Clave foránea hacia Player.
    #
    # `ondelete="CASCADE"` hace que PostgreSQL elimine estas estadísticas
    # si el jugador padre es eliminado.
    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    season: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    games: Mapped[int] = mapped_column(Integer, default=0)
    plate_appearances: Mapped[int] = mapped_column(Integer, default=0)
    hits: Mapped[int] = mapped_column(Integer, default=0)
    home_runs: Mapped[int] = mapped_column(Integer, default=0)
    rbi: Mapped[int] = mapped_column(Integer, default=0)
    walks: Mapped[int] = mapped_column(Integer, default=0)
    strikeouts: Mapped[int] = mapped_column(Integer, default=0)

    # Numeric(5, 3) permite valores como:
    #
    # .285
    # .401
    # 1.025
    #
    # Decimal evita los problemas de precisión inherentes a float
    # cuando queremos representar números decimales exactamente.
    avg: Mapped[Decimal | None] = mapped_column(Numeric(5, 3))
    obp: Mapped[Decimal | None] = mapped_column(Numeric(5, 3))
    slg: Mapped[Decimal | None] = mapped_column(Numeric(5, 3))
    ops: Mapped[Decimal | None] = mapped_column(Numeric(5, 3))
    woba: Mapped[Decimal | None] = mapped_column(Numeric(5, 3))

    # Lado inverso de Player.season_stats.
    player: Mapped[Player] = relationship(
        back_populates="season_stats",
    )

    __table_args__ = (
        # Esta restricción vive en PostgreSQL.
        #
        # Incluso si existe un bug en nuestra API o service, la base de
        # datos impedirá duplicar las estadísticas del mismo jugador
        # para una temporada.
        UniqueConstraint(
            "player_id",
            "season",
            name="uq_player_season",
        ),
    )