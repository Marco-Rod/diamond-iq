from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from app.ai.usage import LLMUsageTracker


@pytest.mark.anyio
async def test_record_call_persists_usage_event() -> None:
    session = AsyncMock()
    repository = AsyncMock()
    cost_calculator = Mock()

    repository.add.side_effect = lambda event: event

    cost_calculator.calculate.return_value = Decimal(
        "0.001234"
    )

    tracker = LLMUsageTracker(
        session=session,
        repository=repository,
        cost_calculator=cost_calculator,
    )

    event = await tracker.record_call(
        conversation_id="conversation-123",
        provider="test-provider",
        model="test-model",
        input_tokens=800,
        output_tokens=200,
        tool_calls_requested=2,
    )

    assert event.provider == "test-provider"
    assert event.model == "test-model"

    assert event.input_tokens == 800
    assert event.output_tokens == 200
    assert event.total_tokens == 1000

    assert event.tool_calls_requested == 2

    assert event.estimated_cost_usd == Decimal(
        "0.001234"
    )

    cost_calculator.calculate.assert_called_once_with(
        provider="test-provider",
        model="test-model",
        input_tokens=800,
        output_tokens=200,
    )

    repository.add.assert_awaited_once()
    session.commit.assert_awaited_once()