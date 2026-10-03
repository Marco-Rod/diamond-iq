from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.llm_usage import LLMUsageEvent


class LLMUsageRepository:
    """
    Encapsula el acceso a datos relacionado con consumo de LLMs.

    Esta capa únicamente persiste eventos. No calcula costos ni decide
    qué información debe registrarse.
    """

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self.session = session

    async def add(
        self,
        event: LLMUsageEvent,
    ) -> LLMUsageEvent:
        self.session.add(event)

        # flush() envía el INSERT a PostgreSQL dentro de la transacción
        # actual, pero no confirma todavía la transacción.
        await self.session.flush()

        return event