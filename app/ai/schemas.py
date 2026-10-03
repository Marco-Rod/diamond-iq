from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Representa una pregunta enviada al asistente de Diamond IQ.

    Por ahora el contrato es deliberadamente pequeño:
    únicamente necesitamos el mensaje del usuario.

    Más adelante podremos añadir campos como:
    - conversation_id;
    - preferred_provider;
    - metadata;
    - user_id.
    """

    message: str = Field(
        min_length=1,
        max_length=2000,
        description="Pregunta o instrucción enviada al asistente.",
        examples=[
            "¿Quiénes son los tres jugadores con mayor Power de Yankees?"
        ],
    )


class UsageInfo(BaseModel):
    """
    Normaliza la información de consumo de tokens.

    Los distintos proveedores pueden devolver usage con formatos
    diferentes. Diamond IQ expondrá un formato común para que las capas
    superiores no dependan de detalles específicos de Ollama u OpenAI.
    """

    input_tokens: int = Field(
        default=0,
        ge=0,
    )
    output_tokens: int = Field(
        default=0,
        ge=0,
    )


class ToolExecution(BaseModel):
    """
    Registro público de una tool ejecutada durante la conversación.

    `error` solamente contiene información segura y de alto nivel.
    Nunca debemos exponer stack traces, SQL o detalles internos.
    """

    name: str
    success: bool = True
    error: str | None = None


class ProviderMetadata(BaseModel):
    """
    Información sobre el proveedor y modelo que generaron la respuesta.

    Mantener esta información separada permite ampliar posteriormente
    metadata como latency, fallback o finish_reason sin ensuciar el
    contrato principal.
    """

    provider: str
    model: str


class ChatResponse(BaseModel):
    """
    Respuesta pública del asistente de Diamond IQ.

    Este schema es independiente del proveedor concreto utilizado.
    Ollama y OpenAI deberán adaptar sus respuestas a este formato común.
    """

    answer: str

    provider: ProviderMetadata

    tools_used: list[ToolExecution] = Field(
        default_factory=list,
    )

    usage: UsageInfo = Field(
        default_factory=UsageInfo,
    )


class LLMRequest(BaseModel):
    """
    Representación interna de una petición hacia un modelo.

    Este contrato no pertenece a la API pública. Se utiliza dentro
    del módulo de IA para comunicarnos con cualquier provider.
    """

    messages: list[dict[str, Any]]

    temperature: float = Field(
        default=0.2,
        ge=0,
        le=2,
    )

    tools: list[dict[str, Any]] = Field(
        default_factory=list,
    )


class LLMToolCall(BaseModel):
    """
    Representación normalizada de una solicitud de tool realizada
    por un modelo.

    Ollama y OpenAI pueden expresar tool calls de forma distinta,
    pero Diamond IQ utilizará este formato común internamente.
    """

    id: str | None = None
    name: str
    arguments: dict[str, Any]


class LLMResponse(BaseModel):
    """
    Respuesta interna normalizada de cualquier proveedor LLM.

    Esta estructura evita propagar objetos específicos de SDKs externos
    hacia AIService, Router o la API.
    """

    content: str | None = None

    provider: str
    model: str

    input_tokens: int = Field(
        default=0,
        ge=0,
    )
    output_tokens: int = Field(
        default=0,
        ge=0,
    )

    finish_reason: str | None = None

    tool_calls: list[LLMToolCall] = Field(
        default_factory=list,
    )