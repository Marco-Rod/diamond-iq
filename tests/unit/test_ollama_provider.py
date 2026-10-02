from app.ai.providers.ollama import OllamaProvider
from app.ai.schemas import (
    LLMRequest,
)


def test_build_payload_creates_ollama_request() -> None:
    """
    Comprueba que nuestro contrato interno LLMRequest se traduzca
    correctamente al formato esperado por Ollama.

    No realizamos ninguna llamada HTTP en este test.
    """

    provider = OllamaProvider(
        base_url="http://localhost:11434",
        model="qwen2.5-coder:7b",
    )

    request = LLMRequest(
        messages=[
            {
                "role": "user",
                "content": "Who has the highest Power?",
            }
        ],
        temperature=0.2,
    )

    payload = provider._build_payload(request)

    assert payload == {
        "model": "qwen2.5-coder:7b",
        "messages": [
            {
                "role": "user",
                "content": "Who has the highest Power?",
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }


def test_build_payload_includes_tools_when_present() -> None:
    """
    Las tools solo deben enviarse a Ollama cuando realmente existen
    dentro del request.
    """

    provider = OllamaProvider(
        base_url="http://localhost:11434",
        model="qwen2.5-coder:7b",
    )

    tool = {
        "type": "function",
        "function": {
            "name": "get_top_players",
            "description": "Returns top players by rating.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }

    request = LLMRequest(
        messages=[
            {
                "role": "user",
                "content": "Who has the highest Power?",
            }
        ],
        tools=[tool],
    )

    payload = provider._build_payload(request)

    assert payload["tools"] == [tool]


def test_normalize_response_converts_ollama_response() -> None:
    """
    Traduce una respuesta típica de Ollama a nuestro contrato
    independiente LLMResponse.
    """

    provider = OllamaProvider(
        base_url="http://localhost:11434",
        model="qwen2.5-coder:7b",
    )

    ollama_response = {
        "model": "qwen2.5-coder:7b",
        "message": {
            "role": "assistant",
            "content": "Baseball is a bat-and-ball sport.",
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 39,
        "eval_count": 37,
    }

    result = provider._normalize_response(
        ollama_response,
    )

    assert result.provider == "ollama"
    assert result.model == "qwen2.5-coder:7b"
    assert result.content == "Baseball is a bat-and-ball sport."

    assert result.input_tokens == 39
    assert result.output_tokens == 37
    assert result.finish_reason == "stop"

    assert result.tool_calls == []


def test_normalize_response_converts_tool_calls() -> None:
    """
    Comprueba que el formato específico de tool calling de Ollama
    se transforme al formato común de Diamond IQ.
    """

    provider = OllamaProvider(
        base_url="http://localhost:11434",
        model="qwen2.5-coder:7b",
    )

    ollama_response = {
        "model": "qwen2.5-coder:7b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_top_players",
                        "arguments": {
                            "team": "NYY",
                            "metric": "power",
                            "limit": 3,
                        },
                    }
                }
            ],
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 50,
        "eval_count": 10,
    }

    result = provider._normalize_response(
        ollama_response,
    )

    assert result.content is None
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]

    assert tool_call.name == "get_top_players"

    assert tool_call.arguments == {
        "team": "NYY",
        "metric": "power",
        "limit": 3,
    }