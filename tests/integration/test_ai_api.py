from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_chat_returns_stable_response_contract() -> None:
    """
    Verifica que el endpoint de IA respete el contrato público
    antes de conectar proveedores reales.
    """

    response = client.post(
        "/api/v1/ai/chat",
        json={
            "message": "¿Quién tiene mayor Power en Yankees?"
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert "answer" in body

    assert body["provider"]["provider"] == "stub"
    assert body["provider"]["model"] == "none"

    assert body["tools_used"] == []

    assert body["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
    }


def test_chat_rejects_empty_message() -> None:
    """
    FastAPI/Pydantic debe rechazar el request antes de ejecutar
    lógica de IA si el mensaje no cumple el contrato.
    """

    response = client.post(
        "/api/v1/ai/chat",
        json={
            "message": ""
        },
    )

    assert response.status_code == 422