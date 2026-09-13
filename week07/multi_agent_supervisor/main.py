from graph import graph


if __name__ == "__main__":
    goal = (
        "Explain the advantages and limitations of visual odometry "
        "for GPS-denied drone navigation."
    )

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

    print("=" * 70)
    print("WEEK 7 — MULTI-AGENT SUPERVISOR")
    print("=" * 70)

    print("\nGoal:")
    print(goal)

    print("\nRunning agent system...\n")

    for state in graph.stream(
        initial_state,
        stream_mode="values",
    ):
        print(
            f"[GRAPH] Iteration: {state.get('iteration', 0)} | "
            f"Next: {state.get('next_agent', '')} | "
            f"Completed: {state.get('completed_agents', [])}"
        )

    print("\n" + "=" * 70)
    print("COMPLETED AGENTS")
    print("=" * 70)

    final_state = graph.invoke(initial_state)

    for agent in final_state["completed_agents"]:
        print(f"- {agent}")

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(final_state["final_answer"])