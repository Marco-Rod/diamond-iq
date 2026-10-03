from decimal import Decimal

from app.ai.costs import LLMCostCalculator, ModelPricing


def test_unknown_pricing_returns_zero_cost() -> None:
    calculator = LLMCostCalculator()

    cost = calculator.calculate(
        provider="ollama",
        model="qwen3:8b",
        input_tokens=2500,
        output_tokens=1200,
    )

    assert cost == Decimal("0")



def test_calculates_cost_using_provider_pricing() -> None:
    calculator = LLMCostCalculator(
        pricing={
            (
                "test-provider",
                "test-model",
            ): ModelPricing(
                input_per_million_tokens_usd=Decimal("2"),
                output_per_million_tokens_usd=Decimal("8"),
            )
        }
    )

    cost = calculator.calculate(
        provider="test-provider",
        model="test-model",
        input_tokens=500_000,
        output_tokens=250_000,
    )

    # Input:
    # 500k / 1M * $2 = $1
    #
    # Output:
    # 250k / 1M * $8 = $2
    #
    # Total = $3
    assert cost == Decimal("3")