from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ModelPricing:
    """
    Precio por un millón de tokens.

    Usamos Decimal porque estamos trabajando con dinero y queremos
    evitar errores de precisión de float.
    """

    input_per_million_tokens_usd: Decimal
    output_per_million_tokens_usd: Decimal


class LLMCostCalculator:
    """
    Calcula el costo estimado de una llamada al LLM.

    El calculator no conoce sesiones SQLAlchemy ni persiste información.
    Recibe provider/model/tokens y devuelve únicamente un Decimal.
    """

    def __init__(
        self,
        pricing: dict[tuple[str, str], ModelPricing] | None = None,
    ) -> None:
        self.pricing = pricing or {}

    def calculate(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Decimal:
        """
        Devuelve el costo estimado en USD.

        Los providers/modelos sin pricing configurado se consideran
        costo cero por ahora. Esto cubre nuestro Ollama local.
        """

        model_pricing = self.pricing.get(
            (provider, model)
        )

        if model_pricing is None:
            return Decimal("0")

        million = Decimal("1000000")

        input_cost = (
            Decimal(input_tokens)
            / million
            * model_pricing.input_per_million_tokens_usd
        )

        output_cost = (
            Decimal(output_tokens)
            / million
            * model_pricing.output_per_million_tokens_usd
        )

        return input_cost + output_cost