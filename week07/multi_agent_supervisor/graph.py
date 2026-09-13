from typing import Literal

from langgraph.graph import StateGraph, START, END

from state import MultiAgentState
from agents import research_agent, analyst_agent, writer_agent
from supervisor import supervisor_agent


MAX_ITERATIONS = 6


def supervisor_route(
    state: MultiAgentState,
) -> Literal["research", "analyst", "writer", "finish"]:

    next_agent = state.get("next_agent", "FINISH")
    iteration = state.get("iteration", 0)
    completed_agents = state.get("completed_agents", [])

    # --------------------------------------------------------------
    # Global safety limit
    # --------------------------------------------------------------

    if iteration >= MAX_ITERATIONS:
        return "finish"

    # --------------------------------------------------------------
    # Never allow an already-completed agent to run again.
    # --------------------------------------------------------------

    if next_agent == "research":
        if "research" not in completed_agents:
            return "research"
        return "finish"

    if next_agent == "analyst":
        if "analyst" not in completed_agents:
            return "analyst"
        return "finish"

    if next_agent == "writer":
        if "writer" not in completed_agents:
            return "writer"
        return "finish"

    return "finish"


def finalize(state: MultiAgentState) -> dict:

    draft = state.get("draft", "").strip()

    if draft:
        final_answer = draft
    else:
        final_answer = (
            "The agent system reached its iteration limit "
            "without producing a final draft."
        )

    return {
        "final_answer": final_answer,
    }


# ------------------------------------------------------------------
# Build LangGraph
# ------------------------------------------------------------------

builder = StateGraph(MultiAgentState)

builder.add_node("supervisor", supervisor_agent)
builder.add_node("research", research_agent)
builder.add_node("analyst", analyst_agent)
builder.add_node("writer", writer_agent)
builder.add_node("finalize", finalize)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    supervisor_route,
    {
        "research": "research",
        "analyst": "analyst",
        "writer": "writer",
        "finish": "finalize",
    },
)

builder.add_edge("research", "supervisor")
builder.add_edge("analyst", "supervisor")
builder.add_edge("writer", "supervisor")

builder.add_edge("finalize", END)


graph = builder.compile()