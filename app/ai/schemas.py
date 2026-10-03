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


class LLMToolCall(BaseModel):
    """
    Representa una llamada a una tool solicitada por un modelo.

    Este schema es neutral respecto al provider. Ollama, Gemini,
    OpenAI u otros providers deben convertir sus formatos nativos
    a esta estructura antes de devolver la respuesta al resto
    de Diamond IQ.
    """

    id: str | None = None
    name: str
    arguments: dict[str, Any] = Field(
        default_factory=dict,
    )

class LLMProviderState(BaseModel):
    """
    Estado opaco perteneciente a un provider específico.

    Diamond IQ puede transportar este estado entre llamadas al LLM,
    pero las capas superiores, como AIService, no deben interpretar
    el contenido de `data`.

    Cada provider decide qué información necesita conservar.

    Ejemplo para Gemini:

        {
            "provider": "gemini",
            "data": {
                "interaction_id": "abc123",
                "pending_tool_calls": {
                    "call_1": "get_player_stats"
                }
            }
        }

    Esto permite conservar información específica del provider sin
    acoplar AIService o LLMRouter a detalles internos de Gemini.
    """

    provider: str

    data: dict[str, Any] = Field(
        default_factory=dict,
    )


class LLMRequest(BaseModel):
    """
    Petición neutral enviada a cualquier provider de LLM.

    Cada adapter es responsable de traducir esta estructura
    al formato particular que requiere su API.
    """

    messages: list[dict[str, Any]]

    temperature: float = 0.2

    tools: list[dict[str, Any]] = Field(
        default_factory=list,
    )

    # Estado opcional utilizado para continuar una conversación
    # cuyo contexto pertenece a un provider específico.
    provider_state: LLMProviderState | None = None



class LLMResponse(BaseModel):
    """
    Respuesta normalizada producida por cualquier provider.

    Gracias a este contrato común, AIService no necesita conocer
    las diferencias entre Ollama, Gemini, OpenAI, etc.
    """

    content: str | None = None

    provider: str

    model: str

    input_tokens: int = 0

    output_tokens: int = 0

    finish_reason: str | None = None

    tool_calls: list[LLMToolCall] = Field(
        default_factory=list,
    )

    # Si el provider necesita conservar contexto específico entre
    # llamadas, puede devolverlo aquí.
    provider_state: LLMProviderState | None = None