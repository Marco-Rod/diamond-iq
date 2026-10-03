from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

PlayerRatingMetric = Literal[
    "contact",
    "power",
    "vision",
    "overall",
]


class GetTopPlayersArguments(BaseModel):
    """
    Argumentos permitidos para la tool `get_top_players`.

    Este schema funciona como frontera de seguridad entre:

    LLM
      ↓
    argumentos generados
      ↓
    Pydantic
      ↓
    backend

    El modelo puede proponer una llamada, pero el backend solo ejecutará
    valores que cumplan este contrato.
    """

    metric: PlayerRatingMetric = Field(
        description=(
            "Diamond IQ rating used to rank players. "
            "Allowed values: contact, power, vision, overall."
        ),
    )

    team: str | None = Field(
        default=None,
        min_length=1,
        max_length=10,
        description=(
            "Optional MLB team abbreviation, for example NYY or LAD."
        ),
    )

    limit: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum number of players to return.",
    )


class TopPlayerResult(BaseModel):
    """
    Representación compacta de un jugador devuelto por la tool.

    No enviamos el objeto ORM completo al LLM. Solo incluimos la
    información necesaria para responder la pregunta.
    """

    id: int
    mlb_id: int
    name: str
    team: str | None = None
    position: str | None = None

    metric: PlayerRatingMetric
    value: int


class GetPlayerArguments(BaseModel):
    """
    Argumentos para buscar jugadores por nombre.

    El nombre no se considera una identidad única. Por ello la tool puede
    devolver más de una coincidencia y dejar que la capa superior maneje
    la ambigüedad.
    """

    player_name: str = Field(
        min_length=1,
        max_length=120,
        description="Player name to search for.",
    )


class PlayerResult(BaseModel):
    """
    Representación compacta de un jugador para consumo del LLM.

    Exponemos únicamente información útil para identificar al jugador y
    responder preguntas generales sobre su perfil.
    """

    id: int
    mlb_id: int
    name: str
    team: str | None = None
    position: str | None = None

    contact: int | None = None
    power: int | None = None
    vision: int | None = None
    overall: int | None = None


class GetPlayerStatsArguments(BaseModel):
    """
    Argumentos para consultar estadísticas de una temporada.
    """

    player_name: str = Field(
        min_length=1,
        max_length=120,
    )

    season: int = Field(
        ge=1900,
        le=2100,
        description="Season year, for example 2026.",
    )


class PlayerStatsResult(BaseModel):
    player_id: int
    name: str
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
        serialization_alias="AVG",
    )
    obp: Decimal | None = Field(
        default=None,
        serialization_alias="OBP",
    )
    slg: Decimal | None = Field(
        default=None,
        serialization_alias="SLG",
    )
    ops: Decimal | None = Field(
        default=None,
        serialization_alias="OPS",
    )
    woba: Decimal | None = Field(
        default=None,
        serialization_alias="wOBA",
    )


class ComparePlayersArguments(BaseModel):
    """
    Argumentos para comparar dos o más jugadores.

    Limitamos la cantidad para controlar cuánto contexto enviamos al LLM.
    """

    player_names: list[str] = Field(
        min_length=2,
        max_length=4,
    )

    metrics: list[PlayerRatingMetric] = Field(
        default_factory=lambda: [
            "contact",
            "power",
            "vision",
            "overall",
        ],
        min_length=1,
        max_length=4,
    )


class PlayerComparisonResult(BaseModel):
    """
    Perfil de ratings utilizado para una comparación.
    """

    id: int
    name: str
    team: str | None = None
    position: str | None = None

    ratings: dict[PlayerRatingMetric, int | None]