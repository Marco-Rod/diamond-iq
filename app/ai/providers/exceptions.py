class LLMProviderError(RuntimeError):
    """
    Error base producido por un provider de LLM.
    """


class LLMProviderTransientError(LLMProviderError):
    """
    Fallo temporal que puede resolverse mediante retry o fallback.

    Ejemplos:
    - timeout;
    - conexión temporalmente caída;
    - HTTP 429;
    - HTTP 5xx.
    """


class LLMProviderPermanentError(LLMProviderError):
    """
    Error que no debería resolverse simplemente repitiendo la petición.

    Ejemplos:
    - request inválido;
    - autenticación incorrecta;
    - modelo inexistente.
    """


class LLMProviderTimeoutError(LLMProviderTransientError):
    """
    El provider no respondió dentro del tiempo permitido.
    """


class LLMProviderUnavailableError(LLMProviderTransientError):
    """
    El provider no pudo ser contactado o está temporalmente indisponible.
    """