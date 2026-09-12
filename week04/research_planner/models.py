from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    id: int
    task: str
    tool: str | None = None
    depends_on: list[int] = Field(default_factory=list)


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep]