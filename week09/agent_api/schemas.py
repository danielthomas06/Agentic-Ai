from typing import Any

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    goal: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The goal or question for the multi-agent system.",
    )


class AgentResponse(BaseModel):
    request_id: str
    goal: str
    answer: str
    agents_used: list[str]
    iterations: int
    execution_time_seconds: float
    success: bool
    trace: list[dict[str, Any]]