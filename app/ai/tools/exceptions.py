class ToolError(RuntimeError):
    """
    Error base para fallos recuperables relacionados con Tool Calling.

    Estas excepciones representan problemas que el LLM puede comprender
    y potencialmente corregir en una iteración posterior.
    """


class ToolArgumentsError(ToolError):
    """
    La tool existe, pero los argumentos no cumplen su contrato.
    """


class ToolNotFoundError(ToolError):
    """
    El modelo intentó ejecutar una tool que no está registrada.
    """


class ToolResultNotFoundError(ToolError):
    """
    La tool fue válida, pero la entidad solicitada no pudo resolverse.

    Ejemplo:
    get_player_stats(player_name="X")
    cuando no existe ningún jugador llamado "X".
    """