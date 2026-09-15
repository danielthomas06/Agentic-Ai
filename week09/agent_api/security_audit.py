import json
import logging
import os
from datetime import datetime, timezone
from typing import Any


AUDIT_LOG_FILE = os.getenv(
    "SECURITY_AUDIT_LOG",
    "logs/security_audit.log",
)


class SecurityAuditLogger:
    """
    Structured security audit logger.

    Security-sensitive events are written as JSON records.
    Secrets and authentication credentials must never be
    included in event metadata.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(
            "security_audit"
        )

        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            os.makedirs(
                os.path.dirname(AUDIT_LOG_FILE) or ".",
                exist_ok=True,
            )

            handler = logging.FileHandler(
                AUDIT_LOG_FILE,
                encoding="utf-8",
            )

            handler.setFormatter(
                logging.Formatter("%(message)s")
            )

            self.logger.addHandler(handler)

    def record(
        self,
        event: str,
        *,
        request_id: str | None = None,
        role: str | None = None,
        capability: str | None = None,
        outcome: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:

        safe_metadata = metadata or {}

        event_record = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "event": event,
            "request_id": request_id,
            "role": role,
            "capability": capability,
            "outcome": outcome,
            "reason": reason,
            "metadata": safe_metadata,
        }

        self.logger.info(
            json.dumps(
                event_record,
                default=str,
            )
        )


audit_logger = SecurityAuditLogger()