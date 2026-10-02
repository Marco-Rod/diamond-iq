from typing import Any

from openai import AsyncOpenAI

from app.ai.schemas import (
    LLMRequest,
    LLMResponse,
    LLMToolCall,
)


class OpenAIProvider:
    """
    Implementación de LLMProvider utilizando OpenAI Responses API.

    Esta clase adapta el contrato interno de Diamond IQ (`LLMRequest`)
    al formato esperado por OpenAI y posteriormente normaliza la
    respuesta externa a nuestro contrato común (`LLMResponse`).

    Las capas superiores no deben depender del SDK de OpenAI.
    """

    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        """
        Configura el cliente async de OpenAI.

        El API key se recibe desde configuración externa para evitar
        secretos hardcodeados dentro del código.
        """

        self.model = model

        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
        )

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Genera una respuesta mediante OpenAI Responses API.

        El método construye el payload específico de OpenAI y luego
        transforma el resultado a `LLMResponse`.
        """

        payload = self._build_payload(request)

        response = await self.client.responses.create(
            **payload,
        )

        return self._normalize_response(response)

    def _build_payload(
        self,
        request: LLMRequest,
    ) -> dict[str, Any]:
        """
        Traduce LLMRequest al formato esperado por Responses API.
        """

        payload: dict[str, Any] = {
            "model": self.model,
            "input": request.messages,
        }

        if request.tools:
            payload["tools"] = request.tools

        return payload

    def _normalize_response(
        self,
        response: Any,
    ) -> LLMResponse:
        """
        Convierte una respuesta de OpenAI a nuestro contrato interno.

        Aquí ocultamos detalles específicos del SDK para que el resto
        de Diamond IQ trabaje únicamente con LLMResponse.
        """

        tool_calls: list[LLMToolCall] = []

        for item in response.output:
            if getattr(item, "type", None) != "function_call":
                continue

            tool_calls.append(
                LLMToolCall(
                    id=getattr(item, "call_id", None),
                    name=item.name,
                    arguments=self._parse_arguments(
                        item.arguments,
                    ),
                )
            )

        usage = getattr(response, "usage", None)

        return LLMResponse(
            content=response.output_text or None,
            provider=self.provider_name,
            model=response.model,
            input_tokens=(
                usage.input_tokens
                if usage is not None
                else 0
            ),
            output_tokens=(
                usage.output_tokens
                if usage is not None
                else 0
            ),
            finish_reason=getattr(
                response,
                "status",
                None,
            ),
            tool_calls=tool_calls,
        )

    def _parse_arguments(
        self,
        arguments: str | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Normaliza argumentos provenientes de function calling.

        Algunos SDKs/proveedores pueden entregarlos como JSON serializado
        y otros como diccionario.
        """

        if isinstance(arguments, dict):
            return arguments

        import json

        return json.loads(arguments)