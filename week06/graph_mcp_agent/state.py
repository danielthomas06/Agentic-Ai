from typing import TypedDict


class AgentState(TypedDict):

    goal: str

    research: str

    analysis: str

    final_answer: str

    messages: list[dict]