import pytest

from app.ai.guardrails import (
    AIGuardrails,
    UnsafeInputError,
    UnsafeOutputError,
)


def test_guardrails_accept_valid_input() -> None:
    guardrails = AIGuardrails(
        max_input_length=100,
    )

    guardrails.validate_input(
        "Who has the highest Power?"
    )


def test_guardrails_reject_empty_input() -> None:
    guardrails = AIGuardrails()

    with pytest.raises(
        UnsafeInputError
    ):
        guardrails.validate_input(
            "   "
        )


def test_guardrails_reject_input_over_limit() -> None:
    guardrails = AIGuardrails(
        max_input_length=5,
    )

    with pytest.raises(
        UnsafeInputError
    ):
        guardrails.validate_input(
            "123456"
        )


def test_guardrails_accept_valid_output() -> None:
    guardrails = AIGuardrails()

    guardrails.validate_output(
        "Mateo Rivera has a Power rating of 98."
    )


def test_guardrails_reject_empty_final_output() -> None:
    guardrails = AIGuardrails()

    with pytest.raises(
        UnsafeOutputError
    ):
        guardrails.validate_output(
            "   "
        )