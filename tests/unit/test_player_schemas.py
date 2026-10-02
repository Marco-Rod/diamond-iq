import pytest
from pydantic import ValidationError

from app.schemas.player import PlayerCreate


def test_player_create_accepts_valid_data() -> None:
    """
    Verifica el caso feliz del schema.

    El objetivo de este test es confirmar que Pydantic acepta un jugador
    cuando todos los campos cumplen las reglas definidas por la API.
    """

    player = PlayerCreate(
        mlb_id=592450,
        name="Aaron Judge",
        team="NYY",
        position="RF",
        contact=85,
        power=99,
        vision=82,
        overall=96,
    )

    assert player.mlb_id == 592450
    assert player.name == "Aaron Judge"
    assert player.power == 99


def test_player_create_allows_missing_optional_ratings() -> None:
    """
    Un jugador puede existir antes de que el pipeline de ratings
    haya calculado sus atributos.

    Por eso los ratings se permiten como None.
    """

    player = PlayerCreate(
        mlb_id=660271,
        name="Shohei Ohtani",
        team="LAD",
        position="DH",
    )

    assert player.contact is None
    assert player.power is None
    assert player.vision is None
    assert player.overall is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("contact", -1),
        ("contact", 101),
        ("power", -1),
        ("power", 101),
        ("vision", -1),
        ("vision", 101),
        ("overall", -1),
        ("overall", 101),
    ],
)
def test_player_create_rejects_ratings_outside_valid_range(
    field: str,
    value: int,
) -> None:
    """
    Todos los ratings de Diamond IQ deben permanecer entre 0 y 100.

    `parametrize` nos permite ejecutar el mismo test con múltiples
    combinaciones sin duplicar código.
    """

    data = {
        "mlb_id": 592450,
        "name": "Aaron Judge",
        "team": "NYY",
        "position": "RF",
        field: value,
    }

    with pytest.raises(ValidationError):
        PlayerCreate(**data)


def test_player_create_rejects_non_positive_mlb_id() -> None:
    """
    MLB utiliza identificadores positivos.

    Rechazamos valores 0 o negativos en la frontera de entrada para
    impedir que información inválida avance hacia service o database.
    """

    with pytest.raises(ValidationError):
        PlayerCreate(
            mlb_id=0,
            name="Invalid Player",
        )


def test_player_create_rejects_empty_name() -> None:
    """
    El nombre tiene `min_length=1`, por lo que una cadena completamente
    vacía no constituye un jugador válido.
    """

    with pytest.raises(ValidationError):
        PlayerCreate(
            mlb_id=123456,
            name="",
        )