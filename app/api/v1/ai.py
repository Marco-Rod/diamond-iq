from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.ai.dependencies import get_llm_router
from app.ai.router import LLMRouter
from app.ai.schemas import ChatRequest, ChatResponse
from app.ai.service import AIService

router = APIRouter(
    prefix="/ai",
    tags=["ai"],
)


def get_ai_service(
    router: Annotated[
        LLMRouter,
        Depends(get_llm_router),
    ],
) -> AIService:
    """
    Construye AIService utilizando el router configurado para el entorno.

    FastAPI resuelve la cadena de dependencias automáticamente:

    get_llm_router
        ↓
    LLMRouter
        ↓
    AIService
        ↓
    endpoint
    """

    return AIService(router)


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
)
async def chat(
    request: ChatRequest,
    service: Annotated[
        AIService,
        Depends(get_ai_service),
    ],
) -> ChatResponse:
    """
    Envía una pregunta al asistente de Diamond IQ.
    """

    return await service.chat(request)