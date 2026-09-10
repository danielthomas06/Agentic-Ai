import json
import requests

from state import AgentState


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"

MAX_STEPS = 10


# ============================================================
# Tools
# ============================================================

def add(a: float, b: float) -> float:
    return a + b


def multiply(a: float, b: float) -> float:
    return a * b


def subtract(a: float, b: float) -> float:
    return a - b


tool_registry = {
    "add": add,
    "multiply": multiply,
    "subtract": subtract,
}


# ============================================================
# Tool definitions
# ============================================================

tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subtract",
            "description": "Subtract b from a.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
]


# ============================================================
# LLM
# ============================================================

def call_llm(state: AgentState):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": state.messages,
            "tools": tool_definitions,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# Tool execution
# ============================================================

def execute_tool(
    state: AgentState,
    tool_name: str,
    arguments: dict,
):

    if tool_name not in tool_registry:

        error = f"Unknown tool: {tool_name}"

        state.record_error(error)

        return {
            "success": False,
            "error": error,
        }

    try:

        function = tool_registry[tool_name]

        result = function(
            arguments["a"],
            arguments["b"],
        )

        state.record_tool(tool_name)

        return {
            "success": True,
            "result": result,
        }

    except Exception as e:

        error = str(e)

        state.record_error(error)

        return {
            "success": False,
            "error": error,
        }


# ============================================================
# Agent
# ============================================================

def run_agent(user_input: str):

    state = AgentState()

    state.add_message(
        {
            "role": "user",
            "content": user_input,
        }
    )

    while state.step < MAX_STEPS:

        state.next_step()

        print(f"\n[STEP {state.step}]")

        response = call_llm(state)

        assistant_message = response["message"]

        tool_calls = assistant_message.get(
            "tool_calls",
            [],
        )

        # ----------------------------------------------------
        # Final answer
        # ----------------------------------------------------

        if not tool_calls:

            answer = assistant_message.get(
                "content",
                "",
            )

            state.set_final_answer(answer)

            return state

        # ----------------------------------------------------
        # Store assistant message
        # ----------------------------------------------------

        state.add_message(
            assistant_message
        )

        # ----------------------------------------------------
        # Execute tools
        # ----------------------------------------------------

        for tool_call in tool_calls:

            function = tool_call["function"]

            tool_name = function["name"]

            arguments = function["arguments"]

            print(
                f"[TOOL] {tool_name}"
                f"({arguments})"
            )

            result = execute_tool(
                state,
                tool_name,
                arguments,
            )

            state.add_message(
                {
                    "role": "tool",
                    "content": json.dumps(result),
                }
            )

    state.set_final_answer(
        "Agent stopped: maximum steps reached."
    )

    return state


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    user_input = input("You: ")

    state = run_agent(user_input)

    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================")

    print(state.final_answer)

    print("\n==============================")
    print("AGENT STATE")
    print("==============================")

    print("Steps:", state.step)
    print("Tools:", state.tools_used)
    print("Errors:", state.errors)