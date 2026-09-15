import re
from dataclasses import dataclass


MAX_OUTPUT_LENGTH = 10000


class OutputGuardrailViolation(ValueError):
    pass


@dataclass
class GuardrailResult:
    safe: bool
    output: str
    reason: str | None = None


SENSITIVE_PATTERNS = [
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*\S+",
    r"\bpassword\s*[:=]\s*\S+",
]


def validate_output(output: str) -> GuardrailResult:
    if not isinstance(output, str):
        return GuardrailResult(
            safe=False,
            output="",
            reason="Output is not a string.",
        )

    output = output.strip()

    if not output:
        return GuardrailResult(
            safe=False,
            output="",
            reason="Output is empty.",
        )

    if len(output) > MAX_OUTPUT_LENGTH:
        return GuardrailResult(
            safe=False,
            output="",
            reason="Output exceeds maximum allowed length.",
        )

    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, output, re.IGNORECASE):
            return GuardrailResult(
                safe=False,
                output="",
                reason="Potential sensitive information detected.",
            )

    return GuardrailResult(
        safe=True,
        output=output,
    )


def enforce_output_guardrails(output: str) -> str:
    result = validate_output(output)

    if not result.safe:
        raise OutputGuardrailViolation(result.reason)

    return result.output