from .evaluator import evaluate_test_case
from .metrics import EvaluationMetrics
from .test_cases import TEST_CASES
from .tracing import Tracer


def run_evaluation() -> list[tuple[EvaluationMetrics, Tracer]]:

    results = []

    print("=" * 70)
    print("WEEK 8 — AGENT EVALUATION")
    print("=" * 70)

    print(f"\nTest cases: {len(TEST_CASES)}")

    for index, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):

        print("\n" + "-" * 70)
        print(
            f"TEST {index}/{len(TEST_CASES)}: "
            f"{test_case.name}"
        )
        print("-" * 70)

        print(f"Goal: {test_case.goal}")

        # evaluate_test_case returns:
        # (metrics, tracer)
        metrics, tracer = evaluate_test_case(test_case)

        results.append((metrics, tracer))

        print(f"\nSuccess: {metrics.success}")
        print(
            f"Latency: "
            f"{metrics.latency_seconds:.2f}s"
        )
        print(
            f"Iterations: "
            f"{metrics.iterations}"
        )
        print(
            f"Agents: "
            f"{metrics.agents_executed}"
        )
        print(
            f"Routing correct: "
            f"{metrics.routing_correct}"
        )
        print(
            f"Answer length: "
            f"{metrics.answer_length}"
        )

        print("\nTrace:")

        for event in tracer.summary():
            print(
                f"  [{event['event_type']}] "
                f"{event['name']} "
                f"{event['duration_seconds']:.3f}s"
            )

        if metrics.error:
            print(
                f"Error: {metrics.error}"
            )

    return results