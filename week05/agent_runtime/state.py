from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    messages: list[dict[str, Any]] = field(
        default_factory=list
    )

    current_step: int = 0

    tool_history: list[str] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    final_answer: str | None = None

    finished: bool = False

    def add_message(
        self,
        role: str,
        content: str,
    ):
        self.messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    def record_tool(
        self,
        tool_name: str,
    ):
        self.tool_history.append(tool_name)

    def record_error(
        self,
        error: str,
    ):
        self.errors.append(error)

    def finish(
        self,
        answer: str,
    ):
        self.final_answer = answer
        self.finished = True