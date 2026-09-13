import time
from typing import Any

from week08.evaluation.traced_graph import build_traced_graph
from week08.evaluation.tracing import Tracer


TRACE_STORE: dict[str, list[dict[str, Any]]] = {}


def run_agent(
    goal: str,
    request_id: str,
) -> dict[str, Any]:
    """
    Run the multi-agent supervisor with full tracing.
    """

    start_time = time.perf_counter()

    tracer = Tracer()
    traced_graph = build_traced_graph(tracer)

    initial_state = {
        "goal": goal,
        "research": "",
        "analysis": "",
        "draft": "",
        "next_agent": "",
        "final_answer": "",
        "completed_agents": [],
        "iteration": 0,
    }

    final_state: dict[str, Any] | None = None

    with tracer.span(
        name="agent_run",
        event_type="run",
        metadata={
            "request_id": request_id,
            "goal": goal,
        },
    ):
        for state in traced_graph.stream(
            initial_state,
            stream_mode="values",
        ):
            final_state = state

    if final_state is None:
        raise RuntimeError("Agent graph returned no final state.")

    execution_time = time.perf_counter() - start_time

    answer = final_state.get("final_answer", "")
    agents_used = final_state.get("completed_agents", [])
    iterations = final_state.get("iteration", 0)

    trace = tracer.summary()

    TRACE_STORE[request_id] = trace

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