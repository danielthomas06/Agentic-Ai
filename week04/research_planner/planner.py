import json
import requests

from models import Plan


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"


PLANNER_PROMPT = """
You are a planning agent.

Convert the user's research goal into a small number of
executable research tasks.

Rules:
1. Create between 3 and 5 steps.
2. Each step must represent a meaningful piece of work.
3. Avoid creating separate steps for minor analysis operations.
4. Research tasks should gather the necessary evidence.
5. Comparison should combine the relevant research results.
6. Recommendation should be based on the comparison.
7. Use dependencies when one step requires another step's result.
8. Return only the structured plan.

Goal:
{goal}
"""

Available tools:
- research: retrieve information from the knowledge base
- analysis: analyze or compare gathered information
- final: prepare the final response

JSON format:

{
    "goal": "string",
    "steps": [
        {
            "id": 1,
            "task": "string",
            "tool": "research",
            "depends_on": []
        }
    ]
}
"""


def create_plan(goal: str) -> Plan:

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": goal,
                },
            ],
            "stream": False,
            "format": "json",
        },
        timeout=300,
    )

    response.raise_for_status()

    content = response.json()["message"]["content"]

    data = json.loads(content)

    return Plan.model_validate(data)