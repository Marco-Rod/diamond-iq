from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import errors

from app.ai.providers.exceptions import (
    LLMProviderPermanentError,
    LLMProviderTransientError,
    LLMProviderUnavailableError,
)
from app.ai.schemas import (
    LLMProviderState,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
)


class GeminiProvider:
    """
    Adaptador entre el contrato neutral de Diamond IQ
    y la API de Interactions de Google Gemini.

    Responsabilidades principales:

    - traducir mensajes internos al formato esperado por Gemini;
    - convertir las definiciones de tools;
    - ejecutar llamadas asíncronas al provider;
    - normalizar respuestas de texto;
    - normalizar function calls;
    - conservar el estado necesario entre interacciones;
    - convertir resultados de tools en function_result;
    - normalizar el consumo de tokens;
    - traducir errores específicos de Gemini a la jerarquía
      común de errores de providers.

    El resto de Diamond IQ no debería necesitar conocer cómo Gemini
    representa internamente sus conversaciones.
    """

    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-3.8-flash",
        timeout_seconds: float = 30.0,
    ) -> None:
        """
        Inicializa el cliente de Gemini.

        La API key se recibe mediante configuración para evitar
        acoplar el provider directamente a Settings.
        """

        self.model = model
        self.timeout_seconds = timeout_seconds

        self.client = genai.Client(
            api_key=api_key,
        )

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Ejecuta o continúa una interacción con Gemini.

        Existen dos escenarios:

        1. Sin provider_state:
           Se inicia una nueva interacción.

        2. Con provider_state perteneciente a Gemini:
           Se continúa la interacción anterior usando
           previous_interaction_id.

        Cuando se continúa una interacción, enviamos como input
        únicamente los resultados de las tools correspondientes
        a la interacción anterior.
        """

        system_instruction = (
            self._extract_system_instruction(
                request,
            )
        )

        tools = self._build_tools(
            request,
        )

        generation_config: dict[str, Any] = {
            "temperature": request.temperature,
        }

        previous_interaction_id = (
            self._get_previous_interaction_id(
                request,
            )
        )

        if previous_interaction_id:
            interaction_input = (
                self._build_continuation_input(
                    request,
                )
            )
            if not interaction_input:
                raise LLMProviderPermanentError(
                    "Gemini continuation requires at least "
                    "one matching tool result."
                )
        else:
            interaction_input = self._build_input(
                request,
            )
        
        try:
            interaction = (
                await self.client.aio.interactions.create(
                    model=self.model,
                    input=interaction_input,
                    previous_interaction_id=(
                        previous_interaction_id
                    ),
                    system_instruction=(
                        system_instruction
                    ),
                    tools=tools or None,
                    generation_config=(
                        generation_config
                    ),
                    timeout=self.timeout_seconds,
                )
            )

        except errors.APIError as exc:
            self._raise_provider_error(exc)
        
        return self._normalize_response(
            interaction,
        )

    def _extract_system_instruction(
        self,
        request: LLMRequest,
    ) -> str | None:
        """
        Extrae los mensajes de sistema de nuestro formato interno.

        Gemini recibe las instrucciones de sistema separadas del
        resto de los mensajes de conversación.
        """

        system_messages = [
            message.get("content", "")
            for message in request.messages
            if message.get("role") == "system"
        ]

        if not system_messages:
            return None

        return "\n\n".join(
            message
            for message in system_messages
            if message
        )

    def _build_input(
        self,
        request: LLMRequest,
    ) -> list[dict[str, Any]]:
        """
        Convierte los mensajes iniciales de Diamond IQ al formato
        de entrada esperado por Gemini.

        En una interacción nueva enviamos únicamente los mensajes
        del usuario.

        Los mensajes de sistema se procesan por separado mediante
        system_instruction.
        """

        inputs: list[dict[str, Any]] = []

        for message in request.messages:
            if message.get("role") != "user":
                continue

            content = message.get(
                "content"
            )

            if not content:
                continue

            inputs.append(
                {
                    "type": "user_input",
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                        }
                    ],
                }
            )

        return inputs

    def _build_tools(
        self,
        request: LLMRequest,
    ) -> list[dict[str, Any]]:
        """
        Convierte nuestras definiciones neutrales de tools
        al formato esperado por Gemini.

        Diamond IQ utiliza internamente una estructura similar a:

            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {...}
                }
            }

        Gemini espera:

            {
                "type": "function",
                "name": "...",
                "description": "...",
                "parameters": {...}
            }

        Esta transformación pertenece al adapter, no al ToolRegistry.
        """

        gemini_tools: list[
            dict[str, Any]
        ] = []

        for tool in request.tools or []:
            function = tool.get(
                "function"
            )

            if not function:
                continue

            gemini_tools.append(
                {
                    "type": "function",
                    "name": function["name"],
                    "description": function.get(
                        "description",
                        "",
                    ),
                    "parameters": function.get(
                        "parameters",
                        {
                            "type": "object",
                            "properties": {},
                        },
                    ),
                }
            )

        return gemini_tools

    def _get_previous_interaction_id(
        self,
        request: LLMRequest,
    ) -> str | None:
        """
        Obtiene el ID de la interacción anterior de Gemini.

        Solo utilizamos el estado cuando pertenece al mismo provider.

        Esto evita que Gemini intente interpretar accidentalmente
        estado generado por Ollama, OpenAI u otro provider.
        """

        state = request.provider_state

        if state is None:
            return None

        if state.provider != self.name:
            return None

        interaction_id = state.data.get(
            "interaction_id"
        )

        if not interaction_id:
            return None

        return str(
            interaction_id
        )

    def _build_continuation_input(
        self,
        request: LLMRequest,
    ) -> list[dict[str, Any]]:
        """
        Convierte resultados internos de tools en function_result.

        Gemini necesita asociar cada resultado con el function_call
        original mediante su call_id.

        `pending_tool_calls` nos permite conocer qué nombre de función
        corresponde a cada tool_call_id producido por la interacción
        anterior.
        """

        state = request.provider_state

        if state is None:
            return []

        pending_tool_calls = (
            state.data.get(
                "pending_tool_calls",
                {},
            )
        )

        results: list[
            dict[str, Any]
        ] = []

        for message in request.messages:
            if message.get("role") != "tool":
                continue

            tool_call_id = message.get(
                "tool_call_id"
            )

            if not tool_call_id:
                continue

            tool_name = (
                pending_tool_calls.get(
                    tool_call_id
                )
            )

            # Ignoramos resultados que no pertenecen a una llamada
            # realizada en la interacción anterior de Gemini.
            if not tool_name:
                continue

            content = message.get(
                "content",
                "",
            )

            # Nuestro contrato normalmente almacena content como texto,
            # pero serializamos defensivamente cualquier otro tipo.
            if not isinstance(
                content,
                str,
            ):
                content = json.dumps(
                    content,
                    default=str,
                )

            results.append(
                {
                    "type": (
                        "function_result"
                    ),
                    "name": tool_name,
                    "call_id": (
                        tool_call_id
                    ),
                    "result": [
                        {
                            "type": "text",
                            "text": content,
                        }
                    ],
                }
            )

        return results

    def _normalize_response(
        self,
        interaction: Any,
    ) -> LLMResponse:
        """
        Convierte una respuesta de Gemini al contrato común
        LLMResponse utilizado por Diamond IQ.

        Además de texto y consumo de tokens, conservamos:

        - function calls;
        - interaction_id;
        - relación call_id -> nombre de tool.

        Esta información permite continuar correctamente una
        interacción después de ejecutar tools.
        """

        tool_calls: list[
            LLMToolCall
        ] = []

        for step in interaction.steps or []:
            if (
                getattr(
                    step,
                    "type",
                    None,
                )
                != "function_call"
            ):
                continue

            tool_calls.append(
                LLMToolCall(
                    id=getattr(
                        step,
                        "id",
                        None,
                    ),
                    name=step.name,
                    arguments=dict(
                        step.arguments
                        or {},
                    ),
                )
            )

        usage = getattr(
            interaction,
            "usage",
            None,
        )

        input_tokens = (
            getattr(
                usage,
                "total_input_tokens",
                0,
            )
            if usage
            else 0
        )

        output_tokens = (
            getattr(
                usage,
                "total_output_tokens",
                0,
            )
            if usage
            else 0
        )

        interaction_id = getattr(
            interaction,
            "id",
            None,
        )

        # Conservamos únicamente las tools generadas en esta interacción.
        # Estos IDs serán necesarios cuando el backend devuelva sus
        # resultados en la siguiente llamada.
        pending_tool_calls = {
            tool_call.id: tool_call.name
            for tool_call in tool_calls
            if tool_call.id
        }

        provider_state = None

        if interaction_id:
            provider_state = (
                LLMProviderState(
                    provider=self.name,
                    data={
                        "interaction_id": (
                            interaction_id
                        ),
                        "pending_tool_calls": (
                            pending_tool_calls
                        ),
                    },
                )
            )
        
        return LLMResponse(
            content=(
                getattr(
                    interaction,
                    "output_text",
                    None,
                )
                or ""
            ),
            provider=self.name,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=(
                output_tokens
            ),
            tool_calls=tool_calls,
            provider_state=(
                provider_state
            ),
        )

    def _raise_provider_error(
        self,
        exc: errors.APIError,
    ) -> None:
        """
        Convierte errores específicos del SDK de Gemini
        a la jerarquía común de Diamond IQ.

        El router utiliza esta clasificación para decidir si debe:

        - reintentar;
        - ejecutar fallback;
        - propagar el error.

        Solo los errores transitorios deberían provocar retries
        automáticos.
        """

        status_code = getattr(
            exc,
            "code",
            None,
        )

        if status_code == 429:
            raise (
                LLMProviderTransientError(
                    "Gemini rate limit exceeded."
                )
            ) from exc

        if (
            status_code is not None
            and status_code >= 500
        ):
            raise (
                LLMProviderUnavailableError(
                    "Gemini is temporarily unavailable."
                )
            ) from exc

        raise LLMProviderPermanentError(
            "Gemini request failed."
        ) from exc