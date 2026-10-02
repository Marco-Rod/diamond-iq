from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.schemas.player import (
    PlayerCreate,
    PlayerDetailResponse,
    PlayerResponse,
)
from app.services.player import (
    PlayerAlreadyExistsError,
    PlayerNotFoundError,
    PlayerService,
)

router = APIRouter(
    prefix="/players",
    tags=["players"],
)


def get_player_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> PlayerService:
    """
    Construye PlayerService usando la sesión creada por FastAPI.

    Esta función actúa como una dependencia intermedia:

    FastAPI crea AsyncSession
            ↓
    get_player_service()
            ↓
    PlayerService
            ↓
    PlayerRepository

    De esta forma los endpoints no necesitan conocer cómo se crea
    una sesión SQLAlchemy ni cómo se construye el repository.
    """
    return PlayerService(session)


@router.post(
    "",
    response_model=PlayerResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_player(
    data: PlayerCreate,
    service: Annotated[
        PlayerService,
        Depends(get_player_service),
    ],
) -> PlayerResponse:
    """
    Crea un nuevo jugador.

    FastAPI valida automáticamente el body utilizando PlayerCreate
    antes de ejecutar este endpoint.

    La API traduce errores del dominio a códigos HTTP, evitando que
    PlayerService dependa directamente de FastAPI.
    """

    try:
        player = await service.create_player(data)

    except PlayerAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return PlayerResponse.model_validate(player)


@router.get(
    "",
    response_model=list[PlayerResponse],
)
async def list_players(
    service: Annotated[
        PlayerService,
        Depends(get_player_service),
    ],
    limit: Annotated[
        int,
        Query(ge=1, le=100),
    ] = 100,
    offset: Annotated[
        int,
        Query(ge=0),
    ] = 0,
) -> list[PlayerResponse]:
    """
    Devuelve una lista paginada de jugadores.

    `limit` restringe la cantidad máxima de registros devueltos.
    Esto evita que un request pueda cargar accidentalmente toda la tabla.
    """

    players = await service.list_players(
        limit=limit,
        offset=offset,
    )

    return [PlayerResponse.model_validate(player) for player in players]


@router.get(
    "/{player_id}",
    response_model=PlayerResponse,
)
async def get_player(
    player_id: int,
    service: Annotated[
        PlayerService,
        Depends(get_player_service),
    ],
) -> PlayerResponse:
    """
    Recupera un jugador mediante su identificador interno.
    """

    try:
        player = await service.get_player(player_id)

    except PlayerNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return PlayerResponse.model_validate(player)


@router.get(
    "/{player_id}/stats",
    response_model=PlayerDetailResponse,
)
async def get_player_stats(
    player_id: int,
    service: Annotated[
        PlayerService,
        Depends(get_player_service),
    ],
) -> PlayerDetailResponse:
    """
    Recupera un jugador junto con todas sus estadísticas disponibles.

    Utilizamos el método especializado `get_player_with_stats()` para
    asegurar que la relación `season_stats` haya sido cargada antes de
    que Pydantic serialice el objeto.
    """

    try:
        player = await service.get_player_with_stats(player_id)

    except PlayerNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return PlayerDetailResponse.model_validate(player)
