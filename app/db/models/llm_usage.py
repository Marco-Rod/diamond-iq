from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Integer, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class LLMUsageEvent(Base):
    """
    Registra el consumo de una llamada individual a un LLM.

    Una misma conversación puede producir varios eventos si el
    orquestador necesita múltiples rondas de tool calling.

    Guardar cada llamada por separado nos permitirá analizar después:
    - consumo por provider;
    - consumo por modelo;
    - cantidad de llamadas por conversación;
    - tool calls solicitadas;
    - costos estimados.
    """

    __tablename__ = "llm_usage_events"

    id: Mapped[int] = mapped_column(
        primary_key=True,
    )

    conversation_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
    )

    provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    model: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
        index=True,
    )

    input_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    output_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    total_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    tool_calls_requested: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    estimated_cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(12, 6),
        nullable=False,
        default=Decimal("0"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )