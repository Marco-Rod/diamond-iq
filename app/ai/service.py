from app.ai.router import LLMRouter
from app.ai.schemas import (
    ChatRequest,
    ChatResponse,
    LLMRequest,
    ProviderMetadata,
    UsageInfo,
)


class AIService:
    """
    Orquesta una interacción con el asistente de Diamond IQ.

    En esta primera versión el service:

    1. recibe ChatRequest;
    2. construye LLMRequest;
    3. delega generación al LLMRouter;
    4. convierte LLMResponse a ChatResponse.

    Tool calling, cuotas y RAG se añadirán posteriormente sin modificar
    el contrato HTTP público.
    """

    def __init__(
        self,
        router: LLMRouter,
    ) -> None:
        self.router = router

    async def chat(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Ejecuta una interacción simple sin tools.

        Por ahora enviamos solamente el mensaje del usuario. Más adelante
        añadiremos system prompts, resultados de tools y contexto RAG.
        """

        llm_request = LLMRequest(
            messages=[
                {
                    "role": "user",
                    "content": request.message,
                }
            ],
            temperature=0.2,
        )

        llm_response = await self.router.generate(
            llm_request,
        )

        return ChatResponse(
            answer=llm_response.content or "",
            provider=ProviderMetadata(
                provider=llm_response.provider,
                model=llm_response.model,
            ),
            usage=UsageInfo(
                input_tokens=llm_response.input_tokens,
                output_tokens=llm_response.output_tokens,
            ),
        )