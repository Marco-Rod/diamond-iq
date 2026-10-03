import json

from app.ai.prompts.system import SYSTEM_PROMPT
from app.ai.router import LLMRouter
from app.ai.schemas import (
    ChatRequest,
    ChatResponse,
    LLMRequest,
    ProviderMetadata,
    ToolExecution,
    UsageInfo,
)
from app.ai.tools.registry import ToolRegistry


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
    ) -> None:
        self.router = router
        self.tool_registry = tool_registry

    async def chat(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Ejecuta una conversación con soporte para una iteración de tools.

        Por ahora soportamos una única ronda de Tool Calling.

        Más adelante, en la fase del orquestador, permitiremos varias
        iteraciones controladas mediante MAX_TOOL_ITERATIONS.
        """

        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": request.message,
            },
        ]

        first_request = LLMRequest(
            messages=messages,
            temperature=0.2,
            tools=self.tool_registry.definitions(),
        )

        first_response = await self.router.generate(
            first_request,
        )

        # Si el modelo responde directamente y no solicita ninguna tool,
        # podemos devolver esa respuesta inmediatamente.
        if not first_response.tool_calls:
            return ChatResponse(
                answer=first_response.content or "",
                provider=ProviderMetadata(
                    provider=first_response.provider,
                    model=first_response.model,
                ),
                usage=UsageInfo(
                    input_tokens=first_response.input_tokens,
                    output_tokens=first_response.output_tokens,
                ),
            )

        tool_executions: list[ToolExecution] = []

        # En esta primera versión manejamos únicamente la primera tool call.
        #
        # Más adelante soportaremos múltiples llamadas e iteraciones
        # utilizando un loop controlado.
        tool_call = first_response.tool_calls[0]

        tool_result = await self.tool_registry.execute(
            name=tool_call.name,
            arguments=tool_call.arguments,
        )

        tool_executions.append(
            ToolExecution(
                name=tool_call.name,
                success=True,
            )
        )

        # Añadimos al historial la decisión tomada por el modelo.
        messages.append(
            {
                "role": "assistant",
                "content": first_response.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        },
                    }
                ],
            }
        )

        # El resultado de la tool se devuelve como datos estructurados.
        #
        # El modelo recibe estos datos como contexto, pero no los obtiene
        # directamente desde PostgreSQL.
        messages.append(
            {
                "role": "tool",
                "content": json.dumps(
                    tool_result,
                    ensure_ascii=False,
                ),
                "name": tool_call.name,
            }
        )

        final_request = LLMRequest(
            messages=messages,
            temperature=0.2,
            # En la segunda llamada no necesitamos volver a ofrecer tools
            # para esta primera implementación de una sola iteración.
            tools=[],
        )

        final_response = await self.router.generate(
            final_request,
        )

        return ChatResponse(
            answer=final_response.content or "",
            provider=ProviderMetadata(
                provider=final_response.provider,
                model=final_response.model,
            ),
            tools_used=tool_executions,
            usage=UsageInfo(
                # Una interacción con tool calling utiliza dos requests al
                # modelo, por lo que sumamos el consumo de ambos.
                input_tokens=(
                    first_response.input_tokens
                    + final_response.input_tokens
                ),
                output_tokens=(
                    first_response.output_tokens
                    + final_response.output_tokens
                ),
            ),
        )