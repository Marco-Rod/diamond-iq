import asyncio
import json

from google import genai

from app.core.config import get_settings


async def main() -> None:
    """
    Comprueba de forma aislada el flujo completo de function calling
    usando Gemini Interactions API.

    Esta prueba no utiliza AIService, LLMRouter ni ToolRegistry.
    """

    settings = get_settings()

    client = genai.Client(
        api_key=settings.gemini_api_key,
    )

    get_player = {
        "type": "function",
        "name": "get_player",
        "description": (
            "Obtiene información de un jugador por nombre."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "player_name": {
                    "type": "string",
                }
            },
            "required": [
                "player_name",
            ],
        },
    }

    first = await client.aio.interactions.create(
        model=settings.gemini_model,
        input="Tell me about Ethan Carter.",
        tools=[
            get_player,
        ],
    )

    print("FIRST ID:")
    print(first.id)

    print("\nFIRST STEPS:")

    function_call = None

    for step in first.steps:
        print(step)

        if step.type == "function_call":
            function_call = step

    if function_call is None:
        print("Gemini did not request a function.")
        return

    fake_result = {
        "id": 1,
        "name": "Ethan Carter",
        "team": "NYY",
        "position": "CF",
    }

    print("\nFUNCTION CALL:")
    print(function_call.name)
    print(function_call.id)
    print(function_call.arguments)

    second = await client.aio.interactions.create(
        model=settings.gemini_model,
        previous_interaction_id=first.id,
        tools=[
            get_player,
        ],
        input=[
            {
                "type": "function_result",
                "name": function_call.name,
                "call_id": function_call.id,
                "result": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            fake_result
                        ),
                    }
                ],
            }
        ],
    )

    print("\nSECOND ID:")
    print(second.id)

    print("\nFINAL RESPONSE:")
    print(second.output_text)


if __name__ == "__main__":
    asyncio.run(main())