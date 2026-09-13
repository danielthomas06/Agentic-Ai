from typing import TypedDict


class MultiAgentState(TypedDict):
    goal: str

    # Agent outputs
    research: str
    analysis: str
    draft: str

    # Supervisor routing
    next_agent: str

    # Final result
    final_answer: str

    # Execution tracking
    completed_agents: list[str]

    # Number of supervisor/worker cycles
    iteration: int