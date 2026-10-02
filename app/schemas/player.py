from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class PlayerBase(BaseModel):
    """
    Campos compartidos por los distintos schemas de Player.

    Esta clase evita duplicar atributos comunes entre los schemas
    utilizados para crear jugadores y para devolverlos desde la API.
    """

    name: str = Field(
        min_length=1,
        max_length=120,
        description="Nombre completo del jugador.",
    )
    team: str | None = Field(
        default=None,
        max_length=80,
        description="Equipo actual o abreviatura del equipo.",
    )
    position: str | None = Field(
        default=None,
        max_length=10,
        description="Posición principal del jugador.",
    )


class PlayerCreate(PlayerBase):
    """
    Datos aceptados por la API cuando queremos crear un jugador.

    `mlb_id` es obligatorio porque representa la identidad externa
    estable del jugador dentro del ecosistema MLB.

    Los ratings son opcionales porque un jugador puede existir en
    Diamond IQ antes de haber sido procesado por el pipeline de ratings.
    """

    mlb_id: int = Field(
        gt=0,
        description="Identificador oficial del jugador en MLB.",
    )

    contact: int | None = Field(
        default=None,
        ge=0,
        le=100,
    )
    power: int | None = Field(
        default=None,
        ge=0,
        le=100,
    )
    vision: int | None = Field(
        default=None,
        ge=0,
        le=100,
    )
    overall: int | None = Field(
        default=None,
        ge=0,
        le=100,
    )


class PlayerResponse(PlayerBase):
    """
    Representación pública de un jugador devuelta por la API.

    Incluye `id` porque este identificador ya existe después de que
    PostgreSQL haya persistido el registro.
    """

    # Permite construir este schema directamente desde objetos ORM
    # de SQLAlchemy usando sus atributos.
    model_config = ConfigDict(from_attributes=True)

    id: int
    mlb_id: int

    contact: int | None = None
    power: int | None = None
    vision: int | None = None
    overall: int | None = None


class PlayerSeasonStatsResponse(BaseModel):
    """
    Estadísticas de un jugador para una temporada concreta.

    Se utiliza como schema de salida; no expone el objeto Player completo
    para evitar relaciones circulares innecesarias.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    player_id: int
    season: int

    games: int
    plate_appearances: int
    hits: int
    home_runs: int
    rbi: int
    walks: int
    strikeouts: int

    avg: Decimal | None = Field(
        default=None,
        ge=Decimal(0),
        le=Decimal(1),
        max_digits=4,
        decimal_places=3,
        examples=[Decimal("0.285")],
    )
    obp: Decimal | None = Field(
        default=None,
        ge=Decimal(0),
        le=Decimal(1),
        max_digits=4,
        decimal_places=3,
        examples=[Decimal("0.372")],
    )
    slg: Decimal | None = Field(
        default=None,
        ge=Decimal(0),
        max_digits=5,
        decimal_places=3,
        examples=[Decimal("0.611")],
    )
    ops:Decimal | None = Field(
        default=None,
        ge=Decimal(0),
        max_digits=4,
        decimal_places=3,
        examples=[Decimal("0.983")],
    )
    woba: Decimal | None = Field(
        default=None,
        ge=Decimal(0),
        le=Decimal(1),
        max_digits=4,
        decimal_places=3,
        examples=[Decimal("0.421")],
    )


class PlayerDetailResponse(PlayerResponse):
    """
    Versión detallada de Player.

    Además de la información básica y ratings, incluye las estadísticas
    disponibles por temporada.
    """

    season_stats: list[PlayerSeasonStatsResponse] = Field(
        default_factory=list,
    )
