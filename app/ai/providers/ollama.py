from typing import Any

import httpx

from app.ai.providers.exceptions import (
    LLMProviderPermanentError,
    LLMProviderTimeoutError,
    LLMProviderUnavailableError,
)
from app.ai.schemas import (
    LLMRequest,
    LLMResponse,
    LLMToolCall,
)


class OllamaProvider:
    """
    Implementación de LLMProvider para un servidor local de Ollama.

    Esta clase adapta el contrato interno de Diamond IQ (`LLMRequest`)
    al formato esperado por Ollama y normaliza posteriormente la respuesta
    externa a nuestro contrato común (`LLMResponse`).

    Ninguna capa superior debería depender directamente del formato JSON
    específico de Ollama.
    """

    provider_name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        """
        Configura la conexión con Ollama.

        `base_url`
            Dirección donde está ejecutándose Ollama.

        `model`
            Modelo que utilizará este provider por defecto.

        `timeout_seconds`
            Tiempo máximo que esperaremos una respuesta antes de considerar
            que el proveedor no respondió correctamente.
        """

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        print(
            "OLLAMA REQUEST:",
            {
                "base_url": self.base_url,
                "model": self.model,
            },
        )
    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Envía una petición de chat a Ollama y devuelve una respuesta
        normalizada.

        `httpx.AsyncClient` permite realizar la llamada HTTP sin bloquear
        el event loop de FastAPI mientras Ollama procesa la petición.
        """

        payload = self._build_payload(request)

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
            ) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )

                response.raise_for_status()

        except httpx.TimeoutException as exc:
            raise LLMProviderTimeoutError(
                f"Ollama timed out after {self.timeout_seconds} seconds."
            ) from exc

        except httpx.ConnectError as exc:
            raise LLMProviderUnavailableError(
                "Could not connect to Ollama."
            ) from exc

        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code

            # Rate limiting o errores del servidor suelen ser temporales.
            if status_code == 429 or status_code >= 500:
                raise LLMProviderUnavailableError(
                    f"Ollama returned transient HTTP {status_code}."
                ) from exc

            # 4xx como modelo inexistente, payload inválido, etc.
            raise LLMProviderPermanentError(
                f"Ollama returned HTTP {status_code}."
            ) from exc

        except httpx.RequestError as exc:
            raise LLMProviderUnavailableError(
                "Ollama request failed."
            ) from exc

        data = response.json()

        return self._normalize_response(data)

    def _build_payload(
        self,
        request: LLMRequest,
    ) -> dict[str, Any]:
        """
        Traduce LLMRequest al formato esperado por Ollama.

        El resto de Diamond IQ no necesita conocer esta estructura.
        """

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": request.messages,
            # Desactivamos streaming inicialmente porque queremos una
            # respuesta completa y sencilla de normalizar.
            "stream": False,
            "options": {
                "temperature": request.temperature,
            },
        }

        if request.tools:
            payload["tools"] = request.tools

        return payload

    def _normalize_response(
        self,
        data: dict[str, Any],
    ) -> LLMResponse:
        """
        Traduce una respuesta específica de Ollama al formato común
        utilizado dentro de Diamond IQ.
        """

        message = data.get("message", {})

        content = message.get("content")

        tool_calls = self._normalize_tool_calls(
            message.get("tool_calls", []),
        )

        return LLMResponse(
            content=content or None,
            provider=self.provider_name,
            model=data.get("model", self.model),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            finish_reason=data.get("done_reason"),
            tool_calls=tool_calls,
        )
    
    def _normalize_tool_calls(
        self,
        raw_tool_calls: list[dict[str, Any]],
    ) -> list[LLMToolCall]:
        """
        Convierte tool calls de Ollama a LLMToolCall.

        En esta fase probablemente recibiremos una lista vacía, pero dejar
        preparada la normalización evita acoplar Tool Calling al formato
        concreto de Ollama cuando lleguemos a la Fase 5.
        """

        normalized_calls: list[LLMToolCall] = []

        for raw_call in raw_tool_calls:
            function = raw_call.get("function", {})

            name = function.get("name")

            # Si el provider devuelve una estructura incompleta, no queremos
            # construir una tool call inválida.
            if not name:
                continue

            normalized_calls.append(
                LLMToolCall(
                    id=raw_call.get("id"),
                    name=name,
                    arguments=function.get("arguments", {}),
                )
            )

        return normalized_calls