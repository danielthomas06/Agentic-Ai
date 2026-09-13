from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import time


@dataclass
class TraceEvent:
    name: str
    event_type: str
    start_time: str
    duration_seconds: float
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class Tracer:
    def __init__(self):
        self.events: list[TraceEvent] = []

    def record(
        self,
        name: str,
        event_type: str,
        start_time: str,
        duration_seconds: float,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ):
        self.events.append(
            TraceEvent(
                name=name,
                event_type=event_type,
                start_time=start_time,
                duration_seconds=duration_seconds,
                success=success,
                metadata=metadata or {},
            )
        )

    def span(
        self,
        name: str,
        event_type: str,
        metadata: dict[str, Any] | None = None,
    ):
        return TraceSpan(
            tracer=self,
            name=name,
            event_type=event_type,
            metadata=metadata or {},
        )

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "name": event.name,
                "event_type": event.event_type,
                "start_time": event.start_time,
                "duration_seconds": round(
                    event.duration_seconds, 3
                ),
                "success": event.success,
                "metadata": event.metadata,
            }
            for event in self.events
        ]


class TraceSpan:
    def __init__(
        self,
        tracer: Tracer,
        name: str,
        event_type: str,
        metadata: dict[str, Any],
    ):
        self.tracer = tracer
        self.name = name
        self.event_type = event_type
        self.metadata = metadata
        self.start = 0.0
        self.start_time = ""
        self.success = True

    def __enter__(self):
        self.start = time.perf_counter()
        self.start_time = datetime.now(
            timezone.utc
        ).isoformat()

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        duration = time.perf_counter() - self.start

        self.success = exc_type is None

        self.tracer.record(
            name=self.name,
            event_type=self.event_type,
            start_time=self.start_time,
            duration_seconds=duration,
            success=self.success,
            metadata=self.metadata,
        )

        return False