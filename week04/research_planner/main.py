from planner import create_plan
from executor import Executor


def main():

    goal = (
        "Research visual odometry and SLAM for "
        "GPS-denied drone navigation, compare their "
        "advantages and disadvantages, and provide "
        "a recommendation."
    )

    print("\nCreating plan...")

    plan = create_plan(goal)

    print("\nPLAN")

    for step in plan.steps:

        print(
            f"{step.id}. "
            f"{step.task} "
            f"(tool={step.tool}, "
            f"depends_on={step.depends_on})"
        )

    print("\nStarting execution...")

    executor = Executor()

    state = executor.run(plan)

    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)

    print("\nSTATUS")

    for step_id, status in state.status.items():

        attempts = state.attempts.get(
            step_id,
            0,
        )

        print(
            f"Step {step_id}: "
            f"{status} "
            f"(attempts={attempts})"
        )

    print("\nRESULTS")

    for step_id, result in state.results.items():

        print("\n" + "-" * 60)

        print(
            f"STEP {step_id}"
        )

        print("-" * 60)

        print(result)

    if state.errors:

        print("\nERROR HISTORY")

        for step_id, error in state.errors.items():

            print(
                f"Step {step_id}: {error}"
            )


if __name__ == "__main__":
    main()