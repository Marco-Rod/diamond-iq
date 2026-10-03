from __future__ import annotations

from pathlib import Path

from app.ai.rag.schemas import (
    KnowledgeChunk,
)


class KnowledgeLoader:
    """
    Carga documentos Markdown y los divide en fragmentos pequeños
    para posteriormente generar embeddings.

    En esta primera versión usamos chunking determinista basado
    en párrafos para mantener el comportamiento simple y fácil
    de inspeccionar.
    """

    def __init__(
        self,
        *,
        knowledge_dir: Path,
        max_chunk_chars: int = 1200,
    ) -> None:
        self.knowledge_dir = knowledge_dir
        self.max_chunk_chars = max_chunk_chars

    def load(self) -> list[KnowledgeChunk]:
        """
        Carga todos los archivos Markdown de la carpeta knowledge.
        """

        chunks: list[KnowledgeChunk] = []

        for path in sorted(
            self.knowledge_dir.glob("*.md")
        ):
            text = path.read_text(
                encoding="utf-8"
            )

            chunks.extend(
                self._chunk_document(
                    source=path.name,
                    text=text,
                )
            )

        return chunks

    def _chunk_document(
        self,
        *,
        source: str,
        text: str,
    ) -> list[KnowledgeChunk]:
        """
        Divide un documento utilizando párrafos como límites
        naturales.

        Evitamos cortar texto arbitrariamente cuando es posible
        para conservar el significado de cada fragmento.
        """

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        chunks: list[KnowledgeChunk] = []
        current_parts: list[str] = []
        current_length = 0
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph_length = len(paragraph)

            if (
                current_parts
                and current_length
                + paragraph_length
                > self.max_chunk_chars
            ):
                chunks.append(
                    KnowledgeChunk(
                        id=f"{source}:{chunk_index}",
                        source=source,
                        content="\n\n".join(
                            current_parts
                        ),
                    )
                )

                chunk_index += 1
                current_parts = []
                current_length = 0

            current_parts.append(
                paragraph
            )

            current_length += (
                paragraph_length
            )

        if current_parts:
            chunks.append(
                KnowledgeChunk(
                    id=f"{source}:{chunk_index}",
                    source=source,
                    content="\n\n".join(
                        current_parts
                    ),
                )
            )

        return chunks