import time

from state import AgentState
from tools import Tool, ToolRegistry
from runtime import AgentRuntime


# ============================================================
# TOOLS
# ============================================================

def calculator(
    a: float,
    b: float,
) -> float:

    return a + b


def multiply(
    a: float,
    b: float,
) -> float:

    return a * b


def slow_tool() -> str:

    time.sleep(10)

    return "Slow tool completed."


# ============================================================
# AGENT
# ============================================================

def demo_agent(
    state: AgentState,
    runtime: AgentRuntime,
) -> bool:

    step = state.current_step

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    if step == 1:

        result = runtime.execute_tool(
            state,
            "calculator",
            a=25,
            b=17,
        )

        state.add_message(
            "tool",
            f"25 + 17 = {result}",
        )

        return False

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    if step == 2:

        result = runtime.execute_tool(
            state,
            "multiply",
            a=42,
            b=3,
        )

        state.add_message(
            "tool",
            f"42 * 3 = {result}",
        )

        return False

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    if step == 3:

        answer = (
            "The calculation is "
            "42 × 3 = 126."
        )

        state.finish(
            answer
        )

        return True

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    registry = ToolRegistry()

    registry.register(
        Tool(
            name="calculator",
            description="Add two numbers.",
            function=calculator,
        )
    )

    registry.register(
        Tool(
            name="multiply",
            description="Multiply two numbers.",
            function=multiply,
        )
    )

    registry.register(
        Tool(
            name="slow_tool",
            description="A deliberately slow tool.",
            function=slow_tool,
        )
    )

    runtime = AgentRuntime(
        tools=registry,
        max_steps=10,
        max_errors=3,
        max_retries=2,
        tool_timeout=5,
    )

    state = runtime.run(
        demo_agent
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(state.final_answer)

    print("\nMESSAGES")

    for message in state.messages:
        print(message)

    print("\nTOOL HISTORY")

    print(state.tool_history)

    print("\nERRORS")

    for error in state.errors:
        print(error)

    print("\nEVENT COUNT")

    print(len(runtime.events))


if __name__ == "__main__":
    main()