from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.costs import LLMCostCalculator
from app.db.models.llm_usage import LLMUsageEvent
from app.repositories.llm_usage import LLMUsageRepository


class LLMUsageTracker:
    """
    Registra el consumo producido por cada llamada individual al LLM.

    Mantener esta lógica fuera de AIService evita mezclar:
    - orquestación del agente;
    - persistencia;
    - accounting de uso.
    """

    def __init__(
        self,
        session: AsyncSession,
        repository: LLMUsageRepository | None = None,
        cost_calculator: LLMCostCalculator | None = None,
    ) -> None:
        self.session = session

        # Permitir inyectar el repository simplifica los tests unitarios.
        self.repository = (
            repository
            or LLMUsageRepository(session)
        )

        self.cost_calculator = (
            cost_calculator
            or LLMCostCalculator()
        )

    async def record_call(
        self,
        *,
        conversation_id: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tool_calls_requested: int,
    ) -> LLMUsageEvent:
        """
        Persiste las métricas de una llamada individual al provider.
        """
        estimated_cost_usd = self.cost_calculator.calculate(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        event = LLMUsageEvent(
            conversation_id=conversation_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=(
                input_tokens
                + output_tokens
            ),
            tool_calls_requested=tool_calls_requested,
            estimated_cost_usd=estimated_cost_usd,
        )

        await self.repository.add(
            event,
        )

        await self.session.commit()

        return event