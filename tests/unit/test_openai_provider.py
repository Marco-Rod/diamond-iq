from app.ai.providers.openai import OpenAIProvider
from app.ai.schemas import LLMRequest


def test_build_payload_creates_openai_request() -> None:
    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-6-luna",
    )

    request = LLMRequest(
        messages=[
            {
                "role": "user",
                "content": "Who has the highest Power?",
            }
        ]
    )

    payload = provider._build_payload(request)

    assert payload == {
        "model": "gpt-6-luna",
        "input": [
            {
                "role": "user",
                "content": "Who has the highest Power?",
            }
        ],
    }

def test_build_payload_includes_tools() -> None:
    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-6-luna",
    )

    tool = {
        "type": "function",
        "name": "get_top_players",
        "description": "Returns top players by rating.",
        "parameters": {
            "type": "object",
            "properties": {},
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