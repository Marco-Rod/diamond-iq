from fastapi import FastAPI

from app.api.v1.players import router as players_router

app = FastAPI(
    title="DiamondIQ",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    """
    Health check básico de la aplicación.

    Por ahora confirma que FastAPI está respondiendo.
    Más adelante podremos decidir si necesitamos un readiness check
    adicional que también compruebe dependencias como PostgreSQL.
    """
    return {"status": "ok"}


app.include_router(
    players_router,
    prefix="/api/v1",
)