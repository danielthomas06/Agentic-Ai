from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    messages: list[dict[str, Any]] = field(default_factory=list)

    step: int = 0

    tools_used: list[str] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)

    final_answer: str | None = None

    def add_message(self, message: dict[str, Any]):
        self.messages.append(message)

    def record_tool(self, tool_name: str):
        self.tools_used.append(tool_name)

    def record_error(self, error: str):
        self.errors.append(error)

    def next_step(self):
        self.step += 1

    def set_final_answer(self, answer: str):
        self.final_answer = answer