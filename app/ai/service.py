import json
from typing import Any
from uuid import uuid4

from app.ai.limits import (
    AIExecutionLimits,
    TokenBudgetExceededError,
    ToolCallLimitError,
    ToolIterationLimitError,
)
from app.ai.prompts.system import SYSTEM_PROMPT
from app.ai.router import LLMRouter
from app.ai.schemas import (
    ChatRequest,
    ChatResponse,
    LLMRequest,
    LLMResponse,
    ProviderMetadata,
    ToolExecution,
    UsageInfo,
)
from app.ai.tools.exceptions import (
    ToolArgumentsError,
    ToolNotFoundError,
    ToolResultNotFoundError,
)
from app.ai.tools.registry import ToolRegistry
from app.ai.usage import LLMUsageTracker

MAX_TOOL_ITERATIONS = 3



class AIService:
    """
    Orquesta una interacción completa con el asistente de Diamond IQ.

    Esta versión soporta el primer flujo de Tool Calling:

    1. envía la pregunta y las tools disponibles al LLM;
    2. detecta si el modelo solicita una tool;
    3. valida y ejecuta la tool mediante ToolRegistry;
    4. devuelve el resultado de la tool al modelo;
    5. obtiene la respuesta final para el usuario.

    El LLM nunca ejecuta directamente lógica de negocio ni SQL.
    """
    
    def __init__(
    self,
    router: LLMRouter,
    tool_registry: ToolRegistry,
    usage_tracker: LLMUsageTracker,
    limits: AIExecutionLimits | None = None,
    ) -> None:
        self.router = router
        self.tool_registry = tool_registry
        self.usage_tracker = usage_tracker
        # El valor por defecto mantiene los tests y usos simples cómodos,
        # pero permite inyectar límites distintos por entorno.
        self.limits = limits or AIExecutionLimits()
    
    def _build_tool_error_message(
    self,
    *,
    tool_name: str,
    error_type: str,
    message: str,
    ) -> dict[str, Any]:
        """
        Construye un resultado de tool recuperable para el LLM.

        El error se envía como datos estructurados para que el modelo
        pueda decidir si debe corregir argumentos, utilizar otra tool
        o explicar que no puede completar la solicitud.

        Nunca debe utilizarse para excepciones internas inesperadas.
        """

        return {
            "role": "tool",
            "content": json.dumps(
                {
                    "ok": False,
                    "error": {
                        "type": error_type,
                        "tool": tool_name,
                        "message": message,
                    },
                },
                ensure_ascii=False,
            ),
        }
    
    def _build_assistant_tool_message(
    self,
    response: LLMResponse,
    ) -> dict[str, Any]:
        """
        Reconstruye el mensaje del assistant que originó las tool calls.

        Este mensaje debe permanecer en el historial para que el modelo
        pueda entender qué herramientas solicitó antes de recibir sus
        resultados.
        """

        tool_calls: list[dict[str, Any]] = []

        for tool_call in response.tool_calls:
            print(
                "TOOL CALL:",
                {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                },
            )
            serialized_call: dict[str, Any] = {
                "function": {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                }
            }

            # Algunos proveedores asignan un identificador a cada tool call.
            # Lo conservamos cuando existe sin obligar a todos los providers
            # a generarlo.
            if tool_call.id is not None:
                serialized_call["id"] = tool_call.id
            
            tool_calls.append(serialized_call)

        return {
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": tool_calls,
        }

    def _build_chat_response(
    self,
    *,
    llm_response: LLMResponse,
    tool_executions: list[ToolExecution],
    input_tokens: int,
    output_tokens: int,
    ) -> ChatResponse:
        """
        Convierte la última respuesta interna del provider en el contrato
        HTTP público utilizado por nuestra API.
        """

        return ChatResponse(
            answer=llm_response.content or "",
            provider=ProviderMetadata(
                provider=llm_response.provider,
                model=llm_response.model,
            ),
            tools_used=tool_executions,
            usage=UsageInfo(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )

    async def chat(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Ejecuta una conversación con Tool Calling iterativo.

        Una iteración representa una ronda en la que el modelo puede
        solicitar una o varias tools.

        Limitamos el número de rondas para evitar que un modelo entre
        accidentalmente en un ciclo infinito de llamadas.
        """
        conversation_id = str(uuid4())

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": request.message,
            },
        ]

        tool_definitions = self.tool_registry.definitions()

        tool_executions: list[ToolExecution] = []

        total_input_tokens = 0
        total_output_tokens = 0
        tool_call_count = 0
        
        # MAX_TOOL_ITERATIONS limita las rondas que pueden ejecutar tools.
        #
        # Permitimos una llamada adicional al LLM después de la última
        # ronda para que pueda producir la respuesta final.
        for iteration in range(
            self.limits.max_tool_iterations + 1
        ):
            llm_request = LLMRequest(
                messages=messages,
                temperature=0.2,
                tools=tool_definitions,
            )

            llm_response = await self.router.generate(
                llm_request,
            )

            await self.usage_tracker.record_call(
                conversation_id=conversation_id,
                provider=llm_response.provider,
                model=llm_response.model,
                input_tokens=llm_response.input_tokens,
                output_tokens=llm_response.output_tokens,
                tool_calls_requested=len(
                    llm_response.tool_calls
                ),
            )
            
            total_input_tokens += llm_response.input_tokens
            total_output_tokens += llm_response.output_tokens
            total_tokens = (
                total_input_tokens
                + total_output_tokens
            )
            tool_call_count += len(
                llm_response.tool_calls
            )

            if (
                tool_call_count
                > self.limits.max_tool_calls_per_chat
            ):
                raise ToolCallLimitError(
                    "The conversation exceeded the maximum "
                    "number of tool calls."
                )
                
            if (
                llm_response.tool_calls
                and total_tokens
                >= self.limits.max_total_tokens_per_chat
            ):
                raise TokenBudgetExceededError(
                    "The conversation exhausted its token budget."
                )
            # Si el modelo ya no solicita ninguna herramienta,
            # consideramos que terminó su razonamiento y devolvemos
            # la respuesta final al cliente.
            if not llm_response.tool_calls:
                return self._build_chat_response(
                    llm_response=llm_response,
                    tool_executions=tool_executions,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                )

            # Si llegamos hasta aquí después de consumir todas las rondas
            # permitidas, no ejecutamos más tools.
            if iteration >= self.limits.max_tool_iterations:
                raise ToolIterationLimitError(
                    "The model exceeded the maximum number "
                    "of tool-calling iterations."
                )

            # Primero registramos en el historial qué tools solicitó
            # el asistente.
            messages.append(
                self._build_assistant_tool_message(
                    llm_response,
                )
            )

            # Una misma respuesta del modelo puede solicitar varias tools.
            for tool_call in llm_response.tool_calls:
                try:
                    tool_result = await self.tool_registry.execute(
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                    )
                    print(
                        f"TOOL RESULT [{tool_call.name}]:",
                        tool_result,
                    )
                except ToolArgumentsError as exc:
                    """
                    El modelo utilizó una tool existente, pero construyó
                    argumentos que no cumplen su contrato.

                    Es un error recuperable: devolvemos la información al LLM
                    para darle oportunidad de corregir su siguiente llamada.
                    """

                    tool_executions.append(
                        ToolExecution(
                            name=tool_call.name,
                            success=False,
                            error="invalid_arguments",
                        )
                    )

                    messages.append(
                        self._build_tool_error_message(
                            tool_name=tool_call.name,
                            error_type="invalid_arguments",
                            message=str(exc),
                        )
                    )

                    continue

                except ToolNotFoundError as exc:
                    """
                    El modelo solicitó una tool que nuestro registry no expone.

                    También es recuperable: podemos indicarle cuáles son sus
                    capacidades reales en lugar de provocar un HTTP 500.
                    """

                    tool_executions.append(
                        ToolExecution(
                            name=tool_call.name,
                            success=False,
                            error="tool_not_found",
                        )
                    )

                    messages.append(
                        self._build_tool_error_message(
                            tool_name=tool_call.name,
                            error_type="tool_not_found",
                            message=str(exc),
                        )
                    )

                    continue

                except ToolResultNotFoundError as exc:
                    """
                    La tool era válida, pero no pudo encontrar la entidad o información
                    solicitada.

                    Se devuelve el error al modelo para permitirle corregir su siguiente
                    acción utilizando el historial disponible.
                    """

                    tool_executions.append(
                        ToolExecution(
                            name=tool_call.name,
                            success=False,
                            error="result_not_found",
                        )
                    )

                    messages.append(
                        self._build_tool_error_message(
                            tool_name=tool_call.name,
                            error_type="result_not_found",
                            message=str(exc),
                        )
                    )

                    continue
                # Cualquier excepción distinta a las anteriores se propaga.
                #
                # Eso es deliberado: no debemos convertir un bug, una caída
                # de PostgreSQL o un error inesperado en "tool not found".
                tool_executions.append(
                    ToolExecution(
                        name=tool_call.name,
                        success=True,
                    )
                )

                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(
                            {
                                "ok": True,
                                "result": tool_result,
                            },
                            ensure_ascii=False,
                        ),
                    }
                )

        # El flujo normal siempre termina dentro del loop.
        # Esta excepción actúa como protección adicional ante cambios
        # futuros en la lógica.
        raise ToolIterationLimitError(
            "Tool orchestration ended unexpectedly."
        )