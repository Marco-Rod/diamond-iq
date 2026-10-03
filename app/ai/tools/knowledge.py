from __future__ import annotations

from app.ai.rag.retriever import (
    KnowledgeRetriever,
)
from app.ai.tools.schemas import (
    KnowledgeSearchResult,
    SearchKnowledgeArguments,
)


async def search_knowledge(
    *,
    arguments: SearchKnowledgeArguments,
    retriever: KnowledgeRetriever,
) -> list[KnowledgeSearchResult]:
    """
    Busca documentación relevante dentro de la base
    de conocimiento de Diamond IQ.

    Esta tool se utiliza para preguntas conceptuales,
    definiciones y metodología.

    No debe utilizarse para consultar estadísticas o ratings
    dinámicos de jugadores, ya que esos datos pertenecen
    a las tools respaldadas por PostgreSQL.
    """

    retrieved = await retriever.search(
        query=arguments.query,
        limit=arguments.limit,
    )

    return [
        KnowledgeSearchResult(
            source=item.chunk.source,
            content=item.chunk.content,
            score=item.score,
        )
        for item in retrieved
    ]