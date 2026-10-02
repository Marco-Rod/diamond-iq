import asyncio

from app.ai.providers.ollama import OllamaProvider
from app.ai.schemas import LLMRequest
from app.core.config import get_settings


async def main() -> None:
    """
    Prueba manual mínima del provider real de Ollama.

    No forma parte de la API ni de la suite automática porque requiere
    que Ollama esté ejecutándose localmente.
    """

    settings = get_settings()

    provider = OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=120.0,
    )

    response = await provider.generate(
        LLMRequest(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Respond in one short sentence: "
                        "what is baseball?"
                    ),
                }
            ],
            temperature=0.2,
        )
    )

    print("Provider:", response.provider)
    print("Model:", response.model)
    print("Content:", response.content)
    print("Input tokens:", response.input_tokens)
    print("Output tokens:", response.output_tokens)
    print("Finish reason:", response.finish_reason)


if __name__ == "__main__":
    asyncio.run(main())