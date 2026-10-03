from __future__ import annotations

import math

from app.ai.rag.schemas import (
    KnowledgeChunk,
    RetrievedChunk,
)


class InMemoryVectorStore:
    """
    Índice vectorial mínimo almacenado en memoria.

    Para Diamond IQ MVP tenemos pocos documentos, por lo que
    recorrer todos los embeddings es suficiente y evita introducir
    infraestructura adicional como pgvector o una vector database.
    """

    def __init__(self) -> None:
        self._items: list[
            tuple[
                KnowledgeChunk,
                list[float],
            ]
        ] = []

    def add(
        self,
        *,
        chunk: KnowledgeChunk,
        embedding: list[float],
    ) -> None:
        self._items.append(
            (
                chunk,
                embedding,
            )
        )

    def search(
        self,
        *,
        query_embedding: list[float],
        limit: int = 3,
    ) -> list[RetrievedChunk]:
        """
        Busca los fragmentos más similares utilizando
        similitud coseno.
        """

        results: list[
            RetrievedChunk
        ] = []

        for chunk, embedding in self._items:
            score = self._cosine_similarity(
                query_embedding,
                embedding,
            )

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                )
            )

        results.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return results[:limit]

    @staticmethod
    def _cosine_similarity(
        left: list[float],
        right: list[float],
    ) -> float:
        """
        Calcula similitud coseno entre dos embeddings.
        """

        if len(left) != len(right):
            raise ValueError(
                "Embedding dimensions do not match."
            )

        dot_product = sum(
            a * b
            for a, b in zip(
                left,
                right,
                strict=True,
            )
        )

        left_norm = math.sqrt(
            sum(
                value * value
                for value in left
            )
        )

        right_norm = math.sqrt(
            sum(
                value * value
                for value in right
            )
        )

        if (
            left_norm == 0
            or right_norm == 0
        ):
            return 0.0

        return (
            dot_product
            / (left_norm * right_norm)
        )