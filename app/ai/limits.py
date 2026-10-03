from dataclasses import dataclass

MAX_TOOL_ITERATIONS = 3
MAX_TOOL_CALLS_PER_CHAT = 6
MAX_TOTAL_TOKENS_PER_CHAT = 12_000


@dataclass(frozen=True)
class AIExecutionLimits:
    """
    Límites de seguridad para una sola ejecución del asistente.

    Estos límites no representan todavía cuotas por usuario.
    Su objetivo es impedir que una conversación individual consuma
    recursos indefinidamente debido a loops o decisiones del modelo.
    """

    max_tool_iterations: int = MAX_TOOL_ITERATIONS
    max_tool_calls_per_chat: int = MAX_TOOL_CALLS_PER_CHAT
    max_total_tokens_per_chat: int = MAX_TOTAL_TOKENS_PER_CHAT


class AIExecutionLimitError(RuntimeError):
    """
    Error base para límites controlados del orquestador.
    """


class ToolIterationLimitError(AIExecutionLimitError):
    """
    Se produce cuando el modelo continúa solicitando tools después
    de alcanzar el máximo de rondas permitido.
    
    El límite evita loops indefinidos entre el LLM y las herramientas.
    """


class ToolCallLimitError(AIExecutionLimitError):
    """
    La conversación intentó ejecutar demasiadas tools.
    """


class TokenBudgetExceededError(AIExecutionLimitError):
    """
    La conversación agotó su presupuesto acumulado de tokens.
    """
