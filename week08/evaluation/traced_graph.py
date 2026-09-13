from typing import Any

from langgraph.graph import StateGraph, START, END

from week07.multi_agent_supervisor.state import MultiAgentState
from week07.multi_agent_supervisor.agents import (
    research_agent,
    analyst_agent,
    writer_agent,
)
from week07.multi_agent_supervisor.supervisor import (
    supervisor_agent,
)

from .tracing import Tracer


MAX_ITERATIONS = 6


def make_traced_node(
    name: str,
    function,
    tracer: Tracer,
):
    """
    Wrap a LangGraph node with a tracing span.
    """

    def traced_node(state: MultiAgentState) -> dict[str, Any]:

        iteration = state.get("iteration", 0)

        with tracer.span(
            name=name,
            event_type="agent",
            metadata={
                "iteration": iteration,
            },
        ):
            return function(state)

    return traced_node


def supervisor_route(
    state: MultiAgentState,
):
    next_agent = state.get(
        "next_agent",
        "FINISH",
    )

    iteration = state.get(
        "iteration",
        0,
    )

    completed_agents = state.get(
        "completed_agents",
        [],
    )

    if iteration >= MAX_ITERATIONS:
        return "finish"

    if next_agent == "research":
        if "research" not in completed_agents:
            return "research"

    if next_agent == "analyst":
        if "analyst" not in completed_agents:
            return "analyst"

    if next_agent == "writer":
        if "writer" not in completed_agents:
            return "writer"

    return "finish"


def finalize(state: MultiAgentState) -> dict:

    draft = state.get(
        "draft",
        "",
    ).strip()

    if draft:
        final_answer = draft
    else:
        final_answer = (
            "The agent system reached its iteration "
            "limit without producing a final draft."
        )

    return {
        "final_answer": final_answer,
    }


def build_traced_graph(
    tracer: Tracer,
):

    builder = StateGraph(MultiAgentState)

    builder.add_node(
        "supervisor",
        make_traced_node(
            "supervisor",
            supervisor_agent,
            tracer,
        ),
    )

    builder.add_node(
        "research",
        make_traced_node(
            "research",
            research_agent,
            tracer,
        ),
    )

    builder.add_node(
        "analyst",
        make_traced_node(
            "analyst",
            analyst_agent,
            tracer,
        ),
    )

    builder.add_node(
        "writer",
        make_traced_node(
            "writer",
            writer_agent,
            tracer,
        ),
    )

    builder.add_node(
        "finalize",
        make_traced_node(
            "finalize",
            finalize,
            tracer,
        ),
    )

    builder.add_edge(
        START,
        "supervisor",
    )

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

    builder.add_edge(
        "research",
        "supervisor",
    )

    builder.add_edge(
        "analyst",
        "supervisor",
    )

    builder.add_edge(
        "writer",
        "supervisor",
    )

    builder.add_edge(
        "finalize",
        END,
    )

    return builder.compile()