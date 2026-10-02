import asyncio
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.db.models import Player, PlayerSeasonStats
from app.db.session import AsyncSessionFactory


# `Path` nos permite localizar el archivo independientemente del sistema
# operativo y evita construir rutas manualmente con "\" o "/".
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEED_FILE = PROJECT_ROOT / "data" / "seed_players.json"


def load_seed_data() -> list[dict[str, Any]]:
    """
    Lee el dataset desde JSON.

    El seed vive fuera del código Python para separar:

    - datos de prueba/demo;
    - lógica encargada de persistir esos datos.

    Esto permite ampliar o modificar el dataset sin tener que cambiar
    el algoritmo de importación.
    """

    with SEED_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def decimal_or_none(
    value: str | None,
) -> Decimal | None:
    """
    Convierte valores decimales provenientes del JSON a Decimal.

    Evitamos convertir primero a float porque podríamos introducir
    pequeñas imprecisiones binarias antes de guardar el valor en un
    NUMERIC de PostgreSQL.
    """

    if value is None:
        return None

    return Decimal(value)


async def seed_players() -> None:
    """
    Inserta los jugadores y sus estadísticas de demostración.

    El seed es seguro de ejecutar más de una vez:

    si encuentra un jugador con el mismo `mlb_id`, lo omite en lugar
    de crear un duplicado.

    Esta primera implementación prioriza simplicidad. No intenta
    sincronizar modificaciones de registros ya existentes.
    """

    seed_data = load_seed_data()

    created_players = 0
    skipped_players = 0

    async with AsyncSessionFactory() as session:
        for player_data in seed_data:
            mlb_id = player_data["mlb_id"]

            # Comprobamos si el jugador ya fue insertado previamente.
            #
            # Usamos mlb_id porque es la identidad externa única y además
            # cuenta con un índice UNIQUE en PostgreSQL.
            statement = select(Player).where(
                Player.mlb_id == mlb_id,
            )

            result = await session.execute(statement)

            existing_player = result.scalar_one_or_none()

            if existing_player is not None:
                skipped_players += 1
                continue

            player = Player(
                mlb_id=mlb_id,
                name=player_data["name"],
                team=player_data.get("team"),
                position=player_data.get("position"),
                contact=player_data.get("contact"),
                power=player_data.get("power"),
                vision=player_data.get("vision"),
                overall=player_data.get("overall"),
            )

            # Construimos las estadísticas y las añadimos a la relación.
            #
            # Debido al cascade configurado en Player.season_stats,
            # SQLAlchemy persistirá también estas entidades cuando
            # agreguemos el Player a la sesión.
            for stats_data in player_data.get(
                "season_stats",
                [],
            ):
                stats = PlayerSeasonStats(
                    season=stats_data["season"],
                    games=stats_data.get("games", 0),
                    plate_appearances=stats_data.get(
                        "plate_appearances",
                        0,
                    ),
                    hits=stats_data.get("hits", 0),
                    home_runs=stats_data.get(
                        "home_runs",
                        0,
                    ),
                    rbi=stats_data.get("rbi", 0),
                    walks=stats_data.get("walks", 0),
                    strikeouts=stats_data.get(
                        "strikeouts",
                        0,
                    ),
                    avg=decimal_or_none(
                        stats_data.get("avg"),
                    ),
                    obp=decimal_or_none(
                        stats_data.get("obp"),
                    ),
                    slg=decimal_or_none(
                        stats_data.get("slg"),
                    ),
                    ops=decimal_or_none(
                        stats_data.get("ops"),
                    ),
                    woba=decimal_or_none(
                        stats_data.get("woba"),
                    ),
                )

                player.season_stats.append(stats)

            session.add(player)

            created_players += 1

        # Confirmamos todas las inserciones como una única transacción.
        #
        # Si ocurriera una excepción antes de este punto, no queremos
        # dejar un seed aplicado parcialmente.
        await session.commit()

    print(
        f"Seed completed: "
        f"{created_players} created, "
        f"{skipped_players} skipped."
    )


if __name__ == "__main__":
    asyncio.run(seed_players())