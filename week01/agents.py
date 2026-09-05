import requests
import json

from pydantic import ValidationError

from tools import tool_definitions, tool_registry


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"

MAX_STEPS = 10


# ============================================================
# Ollama call
# ============================================================

def call_llm(messages):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "tools": tool_definitions,
            "stream": False
        }
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# Execute a tool
# ============================================================

def execute_tool(function_name, arguments):

    # --------------------------------------------------------
    # Find tool
    # --------------------------------------------------------

    tool = tool_registry.get(function_name)

    if tool is None:
        raise ValueError(
            f"Unknown tool requested: {function_name}"
        )

    function = tool["function"]
    argument_model = tool["model"]

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    try:

        validated_arguments = argument_model.model_validate(
            arguments
        )

    except ValidationError as e:

        raise ValueError(
            f"Invalid arguments for {function_name}: {e}"
        )

    # --------------------------------------------------------
    # Execute function
    # --------------------------------------------------------

    result = function(
        validated_arguments.a,
        validated_arguments.b
    )

    return result


# ============================================================
# Agent
# ============================================================

def run_agent(user_input):

    messages = [
        {
            "role": "user",
            "content": user_input
        }
    ]

    # --------------------------------------------------------
    # Agent loop
    # --------------------------------------------------------

    for step in range(1, MAX_STEPS + 1):

        print()
        print("=" * 60)
        print(f"AGENT STEP {step}")
        print("=" * 60)

        # ----------------------------------------------------
        # Ask LLM what to do
        # ----------------------------------------------------

        data = call_llm(messages)

        assistant_message = data["message"]
        print('assistant_message: ', assistant_message)

        # ----------------------------------------------------
        # Check for tool calls
        # ----------------------------------------------------

        tool_calls = assistant_message.get("tool_calls", [])

        # ----------------------------------------------------
        # No tool call → final answer
        # ----------------------------------------------------

        if not tool_calls:

            final_answer = assistant_message.get(
                "content",
                ""
            )

            print("\nFINAL ANSWER")
            print("-" * 60)
            print(final_answer)

            return final_answer

        # ----------------------------------------------------
        # Add assistant message to conversation
        # ----------------------------------------------------

        messages.append(assistant_message)

        # ----------------------------------------------------
        # Execute requested tools
        # ----------------------------------------------------

        for tool_call in tool_calls:

            function_name = tool_call["function"]["name"]

            arguments = tool_call["function"]["arguments"]

            print("\nTool requested:")
            print(function_name)

            print("\nArguments:")
            print(json.dumps(arguments, indent=2))

            try:

                result = execute_tool(
                    function_name,
                    arguments
                )

            except Exception as e:

                print("\nTOOL ERROR:")
                print(e)

                result = f"Tool execution failed: {e}"

            print("\nTool result:")
            print(result)

            # ------------------------------------------------
            # Send tool result back to LLM
            # ------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "content": str(result)
                }
            )

    # --------------------------------------------------------
    # Maximum step protection
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("MAXIMUM STEPS REACHED")
    print("=" * 60)

    return "Agent stopped because maximum steps were reached."


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("WEEK 01 - MULTI-STEP AGENT")
    print("=" * 60)

    user_input = input("\nYou: ")

    run_agent(user_input)
