from dataclasses import dataclass


class AIGuardrailError(RuntimeError):
    """
    Error base producido por una regla de seguridad
    o validación del sistema de IA.
    """


class UnsafeInputError(AIGuardrailError):
    """
    La entrada del usuario viola una regla que impide
    procesarla directamente.
    """


class UnsafeOutputError(AIGuardrailError):
    """
    La respuesta generada por el modelo viola una regla
    que impide devolverla al usuario.
    """


@dataclass(frozen=True)
class AIGuardrails:
    """
    Conjunto de validaciones deterministas aplicadas antes
    y después de interactuar con el modelo.

    Los guardrails no reemplazan al system prompt.

    El prompt orienta al modelo.
    Los guardrails hacen cumplir reglas que no queremos
    dejar únicamente a decisión probabilística del LLM.
    """

    max_input_length: int = 2000

    def validate_input(
        self,
        text: str,
    ) -> None:
        """
        Valida la entrada del usuario antes de enviarla
        al provider.
        """

        normalized = text.strip()

        if not normalized:
            raise UnsafeInputError(
                "The message cannot be empty."
            )

        if len(normalized) > self.max_input_length:
            raise UnsafeInputError(
                "The message exceeds the maximum allowed length."
            )

    def validate_output(
        self,
        text: str | None,
    ) -> None:
        """
        Aplica validaciones mínimas sobre una respuesta final.

        Una respuesta intermedia puede no contener texto cuando
        el modelo únicamente solicita ejecutar una tool, por eso
        None es válido mientras no sea la respuesta final.
        """

        if text is None:
            return

        if not text.strip():
            raise UnsafeOutputError(
                "The model returned an empty response."
            )