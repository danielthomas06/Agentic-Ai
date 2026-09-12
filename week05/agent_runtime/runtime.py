from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from state import AgentState
from events import AgentEvent, create_event
from tools import ToolRegistry


class AgentRuntime:

    def __init__(
        self,
        tools: ToolRegistry | None = None,
        max_steps: int = 10,
        max_errors: int = 3,
        max_retries: int = 2,
        tool_timeout: int = 30,
    ):

        self.tools = tools or ToolRegistry()

        self.max_steps = max_steps
        self.max_errors = max_errors
        self.max_retries = max_retries
        self.tool_timeout = tool_timeout

        self.events: list[AgentEvent] = []

    # ========================================================
    # EVENT HANDLING
    # ========================================================

    def emit(
        self,
        event_type: str,
        **data: Any,
    ):

        event = create_event(
            event_type,
            **data,
        )

        self.events.append(event)

        print(
            f"[EVENT] {event_type}: {data}"
        )

    # ========================================================
    # TOOL EXECUTION
    # ========================================================

    def execute_tool(
        self,
        state: AgentState,
        tool_name: str,
        **kwargs: Any,
    ) -> Any:

        for attempt in range(
            1,
            self.max_retries + 1,
        ):

            self.emit(
                "tool_started",
                tool=tool_name,
                attempt=attempt,
                arguments=kwargs,
            )

            try:

                with ThreadPoolExecutor(
                    max_workers=1
                ) as executor:

                    future = executor.submit(
                        self.tools.execute,
                        tool_name,
                        **kwargs,
                    )

                    result = future.result(
                        timeout=self.tool_timeout
                    )

                state.record_tool(
                    tool_name
                )

                self.emit(
                    "tool_completed",
                    tool=tool_name,
                    attempt=attempt,
                    result=result,
                )

                return result

            except TimeoutError:

                error = (
                    f"Tool '{tool_name}' "
                    f"timed out after "
                    f"{self.tool_timeout} seconds."
                )

                self.emit(
                    "tool_timeout",
                    tool=tool_name,
                    attempt=attempt,
                    error=error,
                )

                state.record_error(
                    error
                )

            except Exception as exc:

                error = str(exc)

                self.emit(
                    "tool_failed",
                    tool=tool_name,
                    attempt=attempt,
                    error=error,
                )

                state.record_error(
                    error
                )

            if attempt < self.max_retries:

                self.emit(
                    "tool_retry",
                    tool=tool_name,
                    next_attempt=attempt + 1,
                )

        raise RuntimeError(
            f"Tool '{tool_name}' failed "
            f"after {self.max_retries} attempts."
        )

    # ========================================================
    # LIMITS
    # ========================================================

    def check_limits(
        self,
        state: AgentState,
    ):

        if state.current_step >= self.max_steps:

            self.emit(
                "runtime_stopped",
                reason="max_steps",
            )

            raise RuntimeError(
                "Maximum agent steps exceeded."
            )

        if len(state.errors) >= self.max_errors:

            self.emit(
                "runtime_stopped",
                reason="max_errors",
            )

            raise RuntimeError(
                "Maximum agent errors exceeded."
            )

    # ========================================================
    # AGENT LOOP
    # ========================================================

    def run(
        self,
        agent: Callable[
            [AgentState, "AgentRuntime"],
            bool
        ],
        state: AgentState | None = None,
    ) -> AgentState:

        if state is None:
            state = AgentState()

        self.emit(
            "agent_started"
        )

        while not state.finished:

            self.check_limits(
                state
            )

            state.current_step += 1

            self.emit(
                "step_started",
                step=state.current_step,
            )

            try:

                finished = agent(
                    state,
                    self,
                )

                if finished:

                    state.finished = True

                    self.emit(
                        "agent_finished",
                        step=state.current_step,
                    )

                else:

                    self.emit(
                        "step_completed",
                        step=state.current_step,
                    )

            except Exception as exc:

                self.emit(
                    "agent_error",
                    step=state.current_step,
                    error=str(exc),
                )

                raise

        return state