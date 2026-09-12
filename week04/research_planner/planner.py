import json
import requests

from models import Plan


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"


SYSTEM_PROMPT = """
You are a planning agent.

Convert the user's research goal into a small number of
executable research tasks.

Rules:
1. Create between 3 and 5 steps.
2. Each step must represent a meaningful piece of work.
3. Avoid separate steps for minor analysis operations.
4. Research tasks should gather necessary evidence.
5. Comparison should combine relevant research results.
6. Recommendation should be based on the comparison.
7. Use dependencies when one step requires another step's result.

Available tools:
- research
- analysis
- final

IMPORTANT:
You MUST return exactly this JSON structure:

{
    "goal": "the original goal",
    "steps": [
        {
            "id": 1,
            "task": "research task",
            "tool": "research",
            "depends_on": []
        }
    ]
}

IMPORTANT:
- Every step MUST have an integer "id".
- Every step MUST have "task".
- Every step MUST have "tool".
- Every step MUST have "depends_on".
- Use "depends_on", NOT "dependencies".
- The top-level object MUST contain "goal".
- IDs must start at 1 and increase sequentially.
- Return ONLY JSON.

Goal:
"""


def create_plan(goal: str) -> Plan:

    prompt = SYSTEM_PROMPT + "\n" + goal

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
            "format": "json",
        },
        timeout=300,
    )

    response.raise_for_status()

    content = response.json()["message"]["content"]

    data = json.loads(content)

    # --------------------------------------------------------
    # Defensive normalization
    # --------------------------------------------------------

    # Ensure the original goal is preserved.
    data["goal"] = goal

    # Normalize the step structure if the model
    # accidentally uses "dependencies".
    for index, step in enumerate(
        data.get("steps", []),
        start=1,
    ):

        if "id" not in step:
            step["id"] = index

        if "depends_on" not in step:

            step["depends_on"] = step.pop(
                "dependencies",
                [],
            )

        if "tool" not in step:

            task = step.get(
                "task",
                ""
            ).lower()

            if "research" in task:
                step["tool"] = "research"

            elif (
                "compare" in task
                or "analy" in task
            ):
                step["tool"] = "analysis"

            else:
                step["tool"] = "final"

    return Plan.model_validate(data)