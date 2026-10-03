from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.dependencies import get_llm_router
from app.ai.limits import AIExecutionLimitError
from app.ai.providers.exceptions import (
    LLMProviderTransientError,
)
from app.ai.router import LLMRouter
from app.ai.schemas import ChatRequest, ChatResponse
from app.ai.service import AIExecutionLimits, AIService
from app.ai.tools.registry import ToolRegistry
from app.ai.usage import LLMUsageTracker
from app.core.config import get_settings
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
    settings = get_settings()

    player_service = PlayerService(session)

    tool_registry = ToolRegistry(
        player_service=player_service,
    )

    usage_tracker = LLMUsageTracker(
        session=session,
    )
    limits = AIExecutionLimits(
        max_tool_iterations=settings.ai_max_tool_iterations,
        max_tool_calls_per_chat=settings.ai_max_tool_calls_per_chat,
        max_total_tokens_per_chat=settings.ai_max_total_tokens_per_chat,
    )

    return AIService(
        router=router,
        tool_registry=tool_registry,
        usage_tracker=usage_tracker,
        limits=limits,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
    service: Annotated[
        AIService,
        Depends(get_ai_service),
    ],
) -> ChatResponse:
    try:
        return await service.chat(request)

    except AIExecutionLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc

    except LLMProviderTransientError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The AI provider is temporarily unavailable."
            ),
        ) from exc