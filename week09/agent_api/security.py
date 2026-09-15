import re

from .security_audit import audit_logger


MAX_GOAL_LENGTH = 2000


class SecurityViolation(ValueError):
    """Raised when an agent request violates a security policy."""


# These are intentionally simple first-line detections.
# They are NOT a complete prompt-injection detector.
INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?previous\s+instructions\b",
    r"\bignore\s+(all\s+)?prior\s+instructions\b",
    r"\bdisregard\s+(all\s+)?previous\s+instructions\b",
    r"\bdisregard\s+(all\s+)?prior\s+instructions\b",
    r"\bforget\s+(all\s+)?previous\s+instructions\b",
    r"\bforget\s+(all\s+)?prior\s+instructions\b",
    r"\b(system|developer)\s+prompt\b",
    r"\breveal\s+(your|the)\s+(system|developer)\s+prompt\b",
    r"\bshow\s+(me\s+)?(your|the)\s+(system|developer)\s+prompt\b",
    r"\bprint\s+(your|the)\s+(system|developer)\s+prompt\b",
    r"\bexecute\s+this\s+command\b",
    r"\brun\s+this\s+command\b",
]


def normalize_goal(goal: str) -> str:
    """
    Normalize user input before security checks and agent execution.
    """

    if not isinstance(goal, str):
        raise SecurityViolation("Goal must be a string.")

    # Remove leading/trailing whitespace.
    goal = goal.strip()

    # Normalize repeated whitespace.
    goal = re.sub(r"\s+", " ", goal)

    return goal


def validate_goal(goal: str) -> str:
    """
    Validate and normalize an agent goal.

    Raises:
        SecurityViolation: if the goal violates security policy.
    """

    # Normalize first.
    goal = normalize_goal(goal)

    # ---------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------
    if not goal:
        raise SecurityViolation("Goal cannot be empty.")

    if len(goal) < 3:
        raise SecurityViolation(
            "Goal must contain at least 3 characters."
        )

    # ---------------------------------------------------------
    # Maximum length protection
    # ---------------------------------------------------------
    if len(goal) > MAX_GOAL_LENGTH:
        audit_logger.record(
            event="input_validation",
            outcome="denied",
            reason="goal_too_long",
            metadata={
                "length": len(goal),
            },
        )

        raise SecurityViolation(
            f"Goal exceeds the maximum length of "
            f"{MAX_GOAL_LENGTH} characters."
        )

    # ---------------------------------------------------------
    # Prompt injection detection
    # ---------------------------------------------------------
    lowered = goal.lower()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            audit_logger.record(
                event="input_validation",
                outcome="denied",
                reason="potential_prompt_injection",
            )

            raise SecurityViolation(
                "The request was rejected by "
                "the security policy."
            )

    # ---------------------------------------------------------
    # Valid request
    # ---------------------------------------------------------
    return goal

