from __future__ import annotations

from app.ai.rag.embeddings import (
    OllamaEmbeddingProvider,
)
from app.ai.rag.loader import (
    KnowledgeLoader,
)
from app.ai.rag.schemas import (
    RetrievedChunk,
)
from app.ai.rag.store import (
    InMemoryVectorStore,
)


class KnowledgeRetriever:
    """
    Orquesta la carga, indexación y recuperación semántica
    de la base de conocimiento de Diamond IQ.
    """

    def __init__(
        self,
        *,
        loader: KnowledgeLoader,
        embeddings: OllamaEmbeddingProvider,
        store: InMemoryVectorStore,
    ) -> None:
        self.loader = loader
        self.embeddings = embeddings
        self.store = store

        self._initialized = False

    async def initialize(self) -> None:
        """
        Genera el índice vectorial una única vez.

        En esta primera versión los embeddings se mantienen
        únicamente en memoria durante la ejecución de la app.
        """

        if self._initialized:
            return

        chunks = self.loader.load()

        if not chunks:
            self._initialized = True
            return

        vectors = await self.embeddings.embed(
            [
                chunk.content
                for chunk in chunks
            ]
        )

        for chunk, vector in zip(
            chunks,
            vectors,
            strict=True,
        ):
            self.store.add(
                chunk=chunk,
                embedding=vector,
            )

        self._initialized = True

    async def search(
        self,
        *,
        query: str,
        limit: int = 3,
    ) -> list[RetrievedChunk]:
        """
        Recupera los fragmentos semánticamente más cercanos
        a una consulta.
        """

        await self.initialize()

        vectors = await self.embeddings.embed(
            [query]
        )

        if not vectors:
            return []

        return self.store.search(
            query_embedding=vectors[0],
            limit=limit,
        )