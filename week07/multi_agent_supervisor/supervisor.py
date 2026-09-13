from typing import Any

import requests

from state import MultiAgentState


MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/chat"
TIMEOUT = 300

VALID_AGENTS = {
    "research",
    "analyst",
    "writer",
    "FINISH",
}


def ask_llm(prompt: str) -> str:
    """Call the local Ollama server through HTTP."""

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"].strip()


def supervisor_agent(state: MultiAgentState) -> dict[str, Any]:
    """
    Supervisor reviews the current state and decides
    which agent should work next.
    """

    goal = state["goal"]
    research = state["research"]
    analysis = state["analysis"]
    draft = state["draft"]
    completed_agents = state["completed_agents"]
    iteration = state["iteration"]

    # --------------------------------------------------------------
    # Determine which agents are actually eligible
    # --------------------------------------------------------------

    available_agents = []

    if "research" not in completed_agents:
        available_agents.append("research")

    if "analyst" not in completed_agents and research.strip():
        available_agents.append("analyst")

    if (
        "writer" not in completed_agents
        and research.strip()
        and analysis.strip()
    ):
        available_agents.append("writer")

    # FINISH is only valid after a draft exists.
    if draft.strip():
        available_agents.append("FINISH")

    # --------------------------------------------------------------
    # Safety: if everything required is complete, finish directly.
    # --------------------------------------------------------------

    if "writer" in completed_agents and draft.strip():
        decision = "FINISH"

        print(f"[SUPERVISOR] Decision: {decision}")

        return {
            "next_agent": decision,
            "iteration": iteration + 1,
        }

    # --------------------------------------------------------------
    # Ask the LLM to choose ONLY from eligible agents.
    # --------------------------------------------------------------

    prompt = f"""
You are the Supervisor of a multi-agent research system.

Your job is to decide which agent should work next.

USER GOAL:
{goal}

CURRENT RESEARCH:
{research}

CURRENT ANALYSIS:
{analysis}

CURRENT DRAFT:
{draft}

COMPLETED AGENTS:
{completed_agents}

CURRENT ITERATION:
{iteration}

ELIGIBLE NEXT AGENTS:
{available_agents}

AGENT ROLES:

research
- Retrieves relevant factual information from the knowledge base.

analyst
- Analyzes the research and identifies technical conclusions,
  trade-offs, advantages, and limitations.

writer
- Produces the final answer using the research and analysis.

FINISH
- Ends the workflow when the final draft is ready.

RULES:

1. Choose ONLY an agent from ELIGIBLE NEXT AGENTS.

2. Never select an agent that has already completed.

3. Research must happen before analysis.

4. Analysis must happen before writing.

5. Writing must happen after research and analysis.

6. If a draft exists, choose FINISH.

7. If the required workflow is complete, choose FINISH.

Return ONLY one value:

research
analyst
writer
FINISH
"""

    decision = ask_llm(prompt)

    decision = decision.strip().lower()

    if decision == "finish":
        decision = "FINISH"

    # --------------------------------------------------------------
    # Validate the LLM decision.
    # --------------------------------------------------------------

    if decision not in VALID_AGENTS:
        raise ValueError(
            f"Invalid supervisor decision: {decision}"
        )

    # --------------------------------------------------------------
    # Critical safety check:
    # LLM cannot bypass workflow constraints.
    # --------------------------------------------------------------

    if decision not in available_agents:
        print(
            f"[SUPERVISOR] Invalid choice '{decision}'. "
            f"Eligible agents: {available_agents}"
        )

        if available_agents:
            decision = available_agents[0]
        else:
            decision = "FINISH"

    print(f"[SUPERVISOR] Decision: {decision}")

    return {
        "next_agent": decision,
        "iteration": iteration + 1,
    }