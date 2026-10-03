from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.dependencies import get_llm_router
from app.ai.router import LLMRouter
from app.ai.schemas import ChatRequest, ChatResponse
from app.ai.service import AIService
from app.ai.tools.registry import ToolRegistry
from app.db.session import get_db_session
from app.services.player import PlayerService

router = APIRouter(
    prefix="/ai",
    tags=["ai"],
)


def get_ai_service(
    router: Annotated[
        LLMRouter,
        Depends(get_llm_router),
    ],
    session: Annotated[
        AsyncSession,
        Depends(get_db_session),
    ],
) -> AIService:
    """
    Construye AIService junto con sus dependencias.

    La misma sesión SQLAlchemy podrá ser reutilizada por todas las tools
    ejecutadas durante esta interacción.
    """

    player_service = PlayerService(session)

    tool_registry = ToolRegistry(
        player_service=player_service,
    )

    return AIService(
        router=router,
        tool_registry=tool_registry,
    )


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