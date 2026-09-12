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
            f"(depends on {step.depends_on})"
        )

    print("\nStarting execution...")

    executor = Executor()

    state = executor.run(plan)

    print("\nEXECUTION COMPLETE")

    for step_id, status in state.status.items():

        print(
            f"Step {step_id}: {status}"
        )

    print("\nRESULTS")

    for step_id, result in state.results.items():

        print(
            f"\nStep {step_id}:"
        )

        print(result)


if __name__ == "__main__":
    main()