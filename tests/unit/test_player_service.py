from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import IntegrityError

from app.db.models import Player
from app.schemas.player import PlayerCreate
from app.services.player import (
    PlayerAlreadyExistsError,
    PlayerNotFoundError,
    PlayerService,
)


@pytest.fixture
def mock_session() -> AsyncMock:
    """
    Crea una sesión async falsa.

    No se conecta a PostgreSQL. Esto permite probar únicamente la lógica
    del service sin depender de infraestructura externa.
    """
    session = AsyncMock()
    return session


@pytest.fixture
def service(mock_session: AsyncMock) -> PlayerService:
    """
    Construye PlayerService utilizando la sesión simulada.

    Después reemplazaremos los métodos del repository interno por mocks
    para controlar exactamente qué comportamiento queremos probar.
    """
    return PlayerService(mock_session)


@pytest.mark.anyio
async def test_get_player_returns_existing_player(
    service: PlayerService,
) -> None:
    """
    Si el repository encuentra el jugador, el service debe devolverlo.
    """

    player = Player(
        id=1,
        mlb_id=592450,
        name="Aaron Judge",
        team="NYY",
        position="RF",
    )

    service.repository.get_by_id = AsyncMock(
        return_value=player,
    )

    result = await service.get_player(1)

    assert result is player

    service.repository.get_by_id.assert_awaited_once_with(1)


@pytest.mark.anyio
async def test_get_player_raises_when_player_does_not_exist(
    service: PlayerService,
) -> None:
    """
    El repository puede devolver None.

    El service traduce ese resultado técnico a un error de negocio:
    PlayerNotFoundError.
    """

    service.repository.get_by_id = AsyncMock(
        return_value=None,
    )

    with pytest.raises(PlayerNotFoundError):
        await service.get_player(999)


@pytest.mark.anyio
async def test_create_player_rejects_existing_mlb_id(
    service: PlayerService,
) -> None:
    """
    No debemos intentar crear un jugador si ya existe otro con el mismo
    identificador MLB.
    """

    existing_player = Player(
        id=1,
        mlb_id=592450,
        name="Aaron Judge",
    )

    service.repository.get_by_mlb_id = AsyncMock(
        return_value=existing_player,
    )

    data = PlayerCreate(
        mlb_id=592450,
        name="Aaron Judge",
        team="NYY",
        position="RF",
    )

    with pytest.raises(PlayerAlreadyExistsError):
        await service.create_player(data)

    service.repository.add = AsyncMock()
    service.repository.add.assert_not_awaited()


@pytest.mark.anyio
async def test_create_player_commits_successful_transaction(
    service: PlayerService,
    mock_session: AsyncMock,
) -> None:
    """
    Una creación exitosa debe:

    1. comprobar que mlb_id no existe;
    2. añadir el jugador;
    3. hacer commit;
    4. refrescar el modelo.
    """

    service.repository.get_by_mlb_id = AsyncMock(
        return_value=None,
    )

    service.repository.add = AsyncMock(
        side_effect=lambda player: player,
    )

    data = PlayerCreate(
        mlb_id=660271,
        name="Shohei Ohtani",
        team="LAD",
        position="DH",
        contact=90,
        power=99,
        vision=88,
        overall=97,
    )

    result = await service.create_player(data)

    assert result.mlb_id == 660271
    assert result.name == "Shohei Ohtani"

    service.repository.add.assert_awaited_once()
    mock_session.commit.assert_awaited_once()
    mock_session.refresh.assert_awaited_once_with(result)


@pytest.mark.anyio
async def test_create_player_rolls_back_on_integrity_error(
    service: PlayerService,
    mock_session: AsyncMock,
) -> None:
    """
    PostgreSQL sigue siendo la última garantía de integridad.

    Si ocurre una violación de constraint durante la creación, el service
    debe revertir la transacción y convertir el error técnico en un error
    comprensible para el dominio.
    """

    service.repository.get_by_mlb_id = AsyncMock(
        return_value=None,
    )

    service.repository.add = AsyncMock(
        side_effect=IntegrityError(
            statement="INSERT",
            params={},
            orig=Exception("duplicate key"),
        ),
    )

    data = PlayerCreate(
        mlb_id=592450,
        name="Aaron Judge",
    )

    with pytest.raises(PlayerAlreadyExistsError):
        await service.create_player(data)

    mock_session.rollback.assert_awaited_once()
    mock_session.commit.assert_not_awaited()