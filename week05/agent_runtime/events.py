from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class AgentEvent:
    event_type: str
    timestamp: str
    data: dict[str, Any]


def create_event(
    event_type: str,
    **data: Any,
) -> AgentEvent:

    return AgentEvent(
        event_type=event_type,
        timestamp=datetime.now().isoformat(),
        data=data,
    )