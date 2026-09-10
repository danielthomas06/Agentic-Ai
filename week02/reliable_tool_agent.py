import json
import time
import requests

from pydantic import BaseModel, ValidationError


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"

MAX_STEPS = 10
MAX_RETRIES = 3


# ============================================================
# Tool argument model
# ============================================================

class TwoNumbers(BaseModel):
    a: float
    b: float


# ============================================================
# Tools
# ============================================================

def add(a: float, b: float) -> float:
    return a + b


def multiply(a: float, b: float) -> float:
    return a * b


def subtract(a: float, b: float) -> float:
    return a - b


# ============================================================
# Tool registry
# ============================================================

tool_registry = {
    "add": {
        "function": add,
        "model": TwoNumbers,
    },
    "multiply": {
        "function": multiply,
        "model": TwoNumbers,
    },
    "subtract": {
        "function": subtract,
        "model": TwoNumbers,
    },
}


# ============================================================
# Ollama tool definitions
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
                    "a": {
                        "type": "number",
                        "description": "First number",
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number",
                    },
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
                    "a": {
                        "type": "number",
                        "description": "First number",
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number",
                    },
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subtract",
            "description": "Subtract the second number from the first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number",
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number",
                    },
                },
                "required": ["a", "b"],
            },
        },
    },
]


# ============================================================
# LLM call
# ============================================================

def call_llm(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tool_definitions,
        "stream": False,
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# Tool execution with validation + retry
# ============================================================

def execute_tool(tool_name, arguments):
    if tool_name not in tool_registry:
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}",
            "retryable": False,
        }

    tool = tool_registry[tool_name]

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    try:
        validated = tool["model"].model_validate(arguments)

    except ValidationError as e:
        return {
            "success": False,
            "error": "Invalid tool arguments",
            "details": e.errors(),
            "retryable": False,
        }

    # --------------------------------------------------------
    # Execute with retry
    # --------------------------------------------------------

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            print(
                f"  [TOOL] {tool_name} "
                f"attempt {attempt}/{MAX_RETRIES}"
            )

            result = tool["function"](
                validated.a,
                validated.b,
            )

            return {
                "success": True,
                "result": result,
                "attempts": attempt,
            }

        except Exception as e:

            print(f"  [ERROR] {e}")

            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue

            return {
                "success": False,
                "error": str(e),
                "attempts": attempt,
                "retryable": True,
            }


# ============================================================
# Agent loop
# ============================================================

def run_agent(user_input):

    messages = [
        {
            "role": "user",
            "content": user_input,
        }
    ]

    for step in range(1, MAX_STEPS + 1):

        print(f"\n[AGENT] Step {step}")

        response = call_llm(messages)

        assistant_message = response["message"]

        tool_calls = assistant_message.get("tool_calls", [])

        # ----------------------------------------------------
        # Final answer
        # ----------------------------------------------------

        if not tool_calls:
            return assistant_message.get(
                "content",
                "",
            )

        # ----------------------------------------------------
        # Add assistant message to conversation state
        # ----------------------------------------------------

        messages.append(assistant_message)

        # ----------------------------------------------------
        # Execute tools
        # ----------------------------------------------------

        for tool_call in tool_calls:

            function = tool_call["function"]

            tool_name = function["name"]
            arguments = function["arguments"]

            print(
                f"[AGENT] Tool call: "
                f"{tool_name}({arguments})"
            )

            result = execute_tool(
                tool_name,
                arguments,
            )

            # Convert result to JSON string
            result_text = json.dumps(result)

            messages.append(
                {
                    "role": "tool",
                    "content": result_text,
                }
            )

    return "Agent stopped: maximum steps reached."


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    user_input = input("You: ")

    answer = run_agent(user_input)

    print("\nAgent:")
    print(answer)