from .runner import run_evaluation


def print_summary(results):

    print("\n")
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    # Each result is:
    # (EvaluationMetrics, Tracer)

    metrics = [
        result[0]
        for result in results
    ]

    total = len(metrics)

    successful = sum(
        result.success
        for result in metrics
    )

    routing_success = sum(
        result.routing_correct
        for result in metrics
    )

    total_latency = sum(
        result.latency_seconds
        for result in metrics
    )

    total_iterations = sum(
        result.iterations
        for result in metrics
    )

    print(f"\nTotal tests:       {total}")
    print(f"Successful:        {successful}")
    print(f"Failed:            {total - successful}")

    if total:
        print(
            f"Success rate:      "
            f"{successful / total * 100:.1f}%"
        )

        print(
            f"Routing accuracy:  "
            f"{routing_success / total * 100:.1f}%"
        )

        print(
            f"Average latency:   "
            f"{total_latency / total:.2f}s"
        )

        print(
            f"Average iterations:"
            f" {total_iterations / total:.1f}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":

    results = run_evaluation()

    print_summary(results)