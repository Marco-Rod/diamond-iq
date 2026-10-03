from pydantic import BaseModel


class KnowledgeChunk(BaseModel):
    """
    Fragmento textual recuperable de la base de conocimiento.

    Cada chunk conserva información sobre su origen para poder
    identificar qué documento respaldó una respuesta.
    """

    id: str
    source: str
    content: str


class RetrievedChunk(BaseModel):
    """
    Resultado de una búsqueda semántica.

    `score` representa qué tan similar es el fragmento respecto
    a la consulta del usuario.
    """

    chunk: KnowledgeChunk
    score: float