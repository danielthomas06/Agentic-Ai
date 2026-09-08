import json
import requests

from config import (
    OLLAMA_URL,
    MODEL,
    MAX_STEPS,
    REQUEST_TIMEOUT,
)

from state import AgentState

from tools import (
    search_wikipedia,
)


# ============================================================
# Tool definitions exposed to the LLM
# ============================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": (
                "Search Wikipedia for a topic and return "
                "a concise factual summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": (
                            "The topic to search for."
                        ),
                    }
                },
                "required": ["topic"],
            },
        },
    }
]


# ============================================================
# Tool registry
# ============================================================

TOOL_REGISTRY = {
    "search_wikipedia": search_wikipedia,
}


# ============================================================
# LLM
# ============================================================

def call_llm(state: AgentState):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": state.messages,
            "tools": TOOL_DEFINITIONS,
            "stream": False,
        },
        timeout=REQUEST_TIMEOUT,
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

    if tool_name not in TOOL_REGISTRY:

        error = f"Unknown tool: {tool_name}"

        state.record_error(error)

        return {
            "success": False,
            "error": error,
        }

    try:

        function = TOOL_REGISTRY[tool_name]

        result = function(**arguments)

        state.record_tool(tool_name)

        return result

    except Exception as e:

        error = str(e)

        state.record_error(error)

        return {
            "success": False,
            "error": error,
        }


# ============================================================
# Agent loop
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

        print(
            f"\n[AGENT] Step {state.step}"
        )

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
        # Store assistant tool call
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

    user_input = input("Research question: ")

    state = run_agent(user_input)

    print("\n================================")
    print("FINAL ANSWER")
    print("================================")

    print(state.final_answer)

    print("\n================================")
    print("AGENT METADATA")
    print("================================")

    print("Steps:", state.step)

    print("Tools used:", state.tools_used)

    print("Errors:", state.errors)