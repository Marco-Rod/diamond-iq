from typing import Annotated

from fastapi import APIRouter, status

from app.ai.schemas import (
    ChatRequest,
    ChatResponse,
    ProviderMetadata,
    UsageInfo,
)

router = APIRouter(
    prefix="/ai",
    tags=["ai"],
)


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
)
async def chat(
    request: ChatRequest,
) -> ChatResponse:
    """
    Endpoint inicial del asistente de Diamond IQ.

    Esta primera versión todavía no llama a Ollama ni OpenAI.

    Su objetivo es validar el contrato HTTP y dejar estable la superficie
    pública de la API antes de introducir providers externos.

    En la siguiente fase, esta implementación temporal será sustituida por
    AIService + LLM Router + providers.
    """

    return ChatResponse(
        answer=(
            "AI provider integration is not enabled yet. "
            f"Received message: {request.message}"
        ),
        provider=ProviderMetadata(
            provider="stub",
            model="none",
        ),
        usage=UsageInfo(),
    )
