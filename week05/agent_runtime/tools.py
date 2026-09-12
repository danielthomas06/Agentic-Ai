from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    function: Callable[..., Any]

    def execute(
        self,
        **kwargs: Any,
    ) -> Any:

        return self.function(**kwargs)


class ToolRegistry:

    def __init__(self):

        self._tools: dict[str, Tool] = {}

    def register(
        self,
        tool: Tool,
    ):

        if tool.name in self._tools:
            raise ValueError(
                f"Tool already registered: {tool.name}"
            )

        self._tools[tool.name] = tool

    def get(
        self,
        name: str,
    ) -> Tool:

        if name not in self._tools:
            raise ValueError(
                f"Unknown tool: {name}"
            )

        return self._tools[name]

    def list_tools(self) -> list[Tool]:

        return list(self._tools.values())

    def execute(
        self,
        name: str,
        **kwargs: Any,
    ) -> Any:

        tool = self.get(name)

        return tool.execute(**kwargs)