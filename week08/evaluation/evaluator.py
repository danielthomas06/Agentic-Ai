import time
from typing import Any

from week07.multi_agent_supervisor.graph import graph

from .metrics import EvaluationMetrics
from .test_cases import TestCase
from .tracing import Tracer


def evaluate_test_case(
    test_case: TestCase,
) -> tuple[EvaluationMetrics, Tracer]:

    initial_state = {
        "goal": test_case.goal,
        "research": "",
        "analysis": "",
        "draft": "",
        "next_agent": "",
        "final_answer": "",
        "completed_agents": [],
        "iteration": 0,
    }

    tracer = Tracer()

    start_time = time.perf_counter()
    final_state: dict[str, Any] | None = None

    try:
        with tracer.span(
            name="agent_run",
            event_type="run",
            metadata={
                "test_name": test_case.name,
                "goal": test_case.goal,
            },
        ):
            for state in graph.stream(
                initial_state,
                stream_mode="values",
            ):
                final_state = state

                tracer.record(
                    name="graph_state",
                    event_type="state",
                    start_time=time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(),
                    ),
                    duration_seconds=0.0,
                    success=True,
                    metadata={
                        "iteration": state.get(
                            "iteration",
                            0,
                        ),
                        "next_agent": state.get(
                            "next_agent",
                            "",
                        ),
                        "completed_agents": state.get(
                            "completed_agents",
                            [],
                        ),
                    },
                )

        elapsed = time.perf_counter() - start_time

        if final_state is None:
            return (
                EvaluationMetrics(
                    test_name=test_case.name,
                    success=False,
                    latency_seconds=elapsed,
                    iterations=0,
                    agents_executed=[],
                    routing_correct=False,
                    answer_length=0,
                    error="Graph returned no state.",
                ),
                tracer,
            )

        agents = final_state.get(
            "completed_agents",
            [],
        )

        answer = final_state.get(
            "final_answer",
            "",
        )

        quality_score, concepts_found = evaluate_answer_quality(
            answer,
            test_case.required_concepts,
        )

        routing_correct = (
            agents == test_case.expected_agents
        )

        success = (
            bool(answer.strip())
            and routing_correct
        )

        metrics = EvaluationMetrics(
            test_name=test_case.name,
            success=success,
            latency_seconds=elapsed,
            iterations=final_state.get(
                "iteration",
                0,
            ),
            agents_executed=agents,
            routing_correct=routing_correct,
            answer_length=len(answer),
            required_concepts=test_case.required_concepts,
            concepts_found=[],
            quality_score=0.0,
        )

        return metrics, tracer

    except Exception as exc:

        elapsed = time.perf_counter() - start_time

        
        tracer.record(
            name="agent_run",
            event_type="error",
            start_time=time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            duration_seconds=elapsed,
            success=False,
            metadata={
                "error": str(exc),
            },
        )

        return (
            EvaluationMetrics(
                test_name=test_case.name,
                success=False,
                latency_seconds=elapsed,
                iterations=0,
                agents_executed=[],
                routing_correct=False,
                answer_length=0,
                error=str(exc),
            ),
            tracer,
        )

def evaluate_answer_quality(
    answer: str,
    required_concepts: list[str],
) -> tuple[float, list[str]]:

    answer_lower = answer.lower()

    found = [
        concept
        for concept in required_concepts
        if concept.lower() in answer_lower
    ]

    if not required_concepts:
        return 1.0, found

    score = len(found) / len(required_concepts)

    return score, found