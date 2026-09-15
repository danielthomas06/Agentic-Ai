import os

from dotenv import load_dotenv
from fastapi import Header, HTTPException

from .security_audit import audit_logger


load_dotenv()


def _load_api_keys() -> dict[str, str]:
    """
    Load API keys from environment variables.
    """

    keys: dict[str, str] = {}

    research_key = os.getenv(
        "AGENT_RESEARCH_API_KEY"
    )

    analyst_key = os.getenv(
        "AGENT_ANALYST_API_KEY"
    )

    writer_key = os.getenv(
        "AGENT_WRITER_API_KEY"
    )

    if research_key:
        keys[research_key] = "research"

    if analyst_key:
        keys[analyst_key] = "analyst"

    if writer_key:
        keys[writer_key] = "writer"

    return keys


def authenticate(
    authorization: str | None = Header(default=None),
) -> str:

    if not authorization:
        audit_logger.record(
            event="authentication",
            outcome="denied",
            reason="missing_credentials",
        )

        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    scheme, _, token = authorization.partition(" ")

    if scheme.lower() != "bearer" or not token:
        audit_logger.record(
            event="authentication",
            outcome="denied",
            reason="invalid_authentication_scheme",
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid authentication scheme.",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    api_keys = _load_api_keys()

    role = api_keys.get(token)

    if role is None:
        audit_logger.record(
            event="authentication",
            outcome="denied",
            reason="invalid_api_key",
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    audit_logger.record(
        event="authentication",
        role=role,
        outcome="allowed",
    )

    return role