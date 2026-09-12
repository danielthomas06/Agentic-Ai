from graph import build_graph


def main():

    goal = (
        "Research visual odometry and SLAM for "
        "GPS-denied drone navigation."
    )

    graph = build_graph()

    initial_state = {
        "goal": goal,
        "research": "",
        "analysis": "",
        "final_answer": "",
        "messages": [],
    }

    print("\nStarting LangGraph agent...")

    result = graph.invoke(
        initial_state
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(
        result["final_answer"]
    )


if __name__ == "__main__":
    main()