from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.api.v1.players import get_player_service
from app.db.models import Player
from app.main import app
from app.services.player import PlayerService

client = TestClient(app)


def test_create_player_returns_201() -> None:
    """
    Verifica el contrato HTTP del endpoint de creación.

    En este test reemplazamos PlayerService por un mock para comprobar
    específicamente la integración entre:

    HTTP
      ↓
    FastAPI
      ↓
    Pydantic
      ↓
    endpoint

    sin depender todavía de PostgreSQL.
    """

    mock_service = AsyncMock(spec=PlayerService)

    player = Player(
        id=1,
        mlb_id=592450,
        name="Aaron Judge",
        team="NYY",
        position="RF",
        contact=85,
        power=99,
        vision=82,
        overall=96,
    )

    mock_service.create_player.return_value = player

    async def override_player_service() -> PlayerService:
        return mock_service

    app.dependency_overrides[get_player_service] = override_player_service

    try:
        response = client.post(
            "/api/v1/players",
            json={
                "mlb_id": 592450,
                "name": "Aaron Judge",
                "team": "NYY",
                "position": "RF",
                "contact": 85,
                "power": 99,
                "vision": 82,
                "overall": 96,
            },
        )

        assert response.status_code == 201

        body = response.json()

        assert body["id"] == 1
        assert body["mlb_id"] == 592450
        assert body["name"] == "Aaron Judge"
        assert body["power"] == 99

        mock_service.create_player.assert_awaited_once()

    finally:
        # Es importante limpiar los overrides para que este test no afecte
        # a los tests siguientes.
        app.dependency_overrides.clear()


def test_create_player_rejects_invalid_rating() -> None:
    """
    FastAPI debe rechazar el request antes de llamar al service
    cuando Pydantic detecta un rating fuera del rango permitido.
    """

    mock_service = AsyncMock(spec=PlayerService)

    async def override_player_service() -> PlayerService:
        return mock_service

    app.dependency_overrides[get_player_service] = override_player_service

    try:
        response = client.post(
            "/api/v1/players",
            json={
                "mlb_id": 592450,
                "name": "Aaron Judge",
                "power": 101,
            },
        )

        assert response.status_code == 422

        mock_service.create_player.assert_not_awaited()

    finally:
        app.dependency_overrides.clear()


def test_list_players_returns_players() -> None:
    """
    Verifica que el endpoint pueda serializar una colección
    de objetos ORM a PlayerResponse.
    """

    mock_service = AsyncMock(spec=PlayerService)

    mock_service.list_players.return_value = [
        Player(
            id=1,
            mlb_id=592450,
            name="Aaron Judge",
            team="NYY",
            position="RF",
            contact=85,
            power=99,
            vision=82,
            overall=96,
        ),
        Player(
            id=2,
            mlb_id=660271,
            name="Shohei Ohtani",
            team="LAD",
            position="DH",
            contact=90,
            power=99,
            vision=88,
            overall=97,
        ),
    ]

    async def override_player_service() -> PlayerService:
        return mock_service

    app.dependency_overrides[get_player_service] = override_player_service

    try:
        response = client.get("/api/v1/players")

        assert response.status_code == 200

        body = response.json()

        assert len(body) == 2
        assert body[0]["name"] == "Aaron Judge"
        assert body[1]["name"] == "Shohei Ohtani"

        mock_service.list_players.assert_awaited_once_with(
            limit=100,
            offset=0,
        )

    finally:
        app.dependency_overrides.clear()
