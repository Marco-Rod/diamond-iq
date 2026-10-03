import asyncio
from pathlib import Path

from app.ai.rag.embeddings import (
    OllamaEmbeddingProvider,
)
from app.ai.rag.loader import (
    KnowledgeLoader,
)
from app.ai.rag.retriever import (
    KnowledgeRetriever,
)
from app.ai.rag.store import (
    InMemoryVectorStore,
)
from app.core.config import get_settings


async def main() -> None:
    """
    Prueba manual del pipeline de recuperación.

    Todavía no interviene ningún LLM:
    únicamente verificamos que una consulta encuentre
    los fragmentos adecuados.
    """

    settings = get_settings()

    retriever = KnowledgeRetriever(
        loader=KnowledgeLoader(
            knowledge_dir=Path(
                "knowledge"
            ),
        ),
        embeddings=OllamaEmbeddingProvider(
            base_url=(
                settings.ollama_base_url
            ),
            model=(
                settings.ollama_embedding_model
            ),
            timeout_seconds=(
                settings.ollama_embedding_timeout_seconds
            ),
        ),
        store=InMemoryVectorStore(),
    )

    results = await retriever.search(
        query=(
            "What does Power mean "
            "in Diamond IQ?"
        ),
        limit=3,
    )

    for result in results:
        print(
            "\nSOURCE:",
            result.chunk.source,
        )

        print(
            "SCORE:",
            result.score,
        )

        print(
            "CONTENT:",
            result.chunk.content,
        )


if __name__ == "__main__":
    asyncio.run(main())