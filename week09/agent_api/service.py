import time
from typing import Any

from week08.evaluation.traced_graph import build_traced_graph
from week08.evaluation.tracing import Tracer

from .output_guardrails import enforce_output_guardrails
from .security import validate_goal


TRACE_STORE: dict[str, list[dict[str, Any]]] = {}


def run_agent(
    goal: str,
    request_id: str,
    agent_role: str,
) -> dict[str, Any]:
    """
    Run the multi-agent supervisor with full tracing.

    Security validation is performed before the agent graph
    is constructed or executed.
    """

    # ---------------------------------------------------------
    # 1. Input security boundary
    # ---------------------------------------------------------
    goal = validate_goal(goal)

    start_time = time.perf_counter()

    # ---------------------------------------------------------
    # 2. Build traced agent graph
    # ---------------------------------------------------------
    tracer = Tracer()
    traced_graph = build_traced_graph(tracer)

    # ---------------------------------------------------------
    # 3. Initial agent state
    # ---------------------------------------------------------
    initial_state = {
        "goal": goal,
        "research": "",
        "analysis": "",
        "draft": "",
        "next_agent": "",
        "final_answer": "",
        "completed_agents": [],
        "iteration": 0,
        "authenticated_role": agent_role,
    }

    final_state: dict[str, Any] | None = None

    # ---------------------------------------------------------
    # 4. Execute agent graph with tracing
    # ---------------------------------------------------------
    with tracer.span(
        name="agent_run",
        event_type="run",
        metadata={
            "request_id": request_id,
            "goal": goal,
            "authenticated_role": agent_role,
        },
    ):
        for state in traced_graph.stream(
            initial_state,
            stream_mode="values",
        ):
            final_state = state

    # ---------------------------------------------------------
    # 5. Validate graph execution
    # ---------------------------------------------------------
    if final_state is None:
        raise RuntimeError("Agent graph returned no final state.")

    execution_time = time.perf_counter() - start_time

    # ---------------------------------------------------------
    # 6. Extract final state
    # ---------------------------------------------------------
    answer = final_state.get("final_answer", "")
    agents_used = final_state.get("completed_agents", [])
    iterations = final_state.get("iteration", 0)

    # ---------------------------------------------------------
    # 7. Output security boundary
    #
    # The final answer must pass the output guardrails
    # before it is returned to the API client.
    # ---------------------------------------------------------
    answer = enforce_output_guardrails(answer)

    # ---------------------------------------------------------
    # 8. Collect and store trace
    # ---------------------------------------------------------
    trace = tracer.summary()

    TRACE_STORE[request_id] = trace

    # ---------------------------------------------------------
    # 9. Return API response
    # ---------------------------------------------------------
    return {
        "request_id": request_id,
        "goal": goal,
        "answer": answer,
        "agents_used": agents_used,
        "iterations": iterations,
        "execution_time_seconds": round(execution_time, 3),
        "success": bool(answer.strip()),
        "trace": trace,
    }


def get_trace(request_id: str) -> list[dict[str, Any]] | None:
    """
    Retrieve a previously stored trace.
    """

    return TRACE_STORE.get(request_id)

