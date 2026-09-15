from week09.agent_api.permissions import (
    AgentRole,
    PermissionDenied,
    authorize,
    is_allowed,
)


def main() -> None:
    # Allowed.
    assert is_allowed(
        AgentRole.RESEARCH,
        "search_knowledge_base",
    )

    assert is_allowed(
        AgentRole.ANALYST,
        "compare_methods",
    )

    # Denied.
    assert not is_allowed(
        AgentRole.WRITER,
        "search_knowledge_base",
    )

    assert not is_allowed(
        AgentRole.RESEARCH,
        "compare_methods",
    )

    # Enforcement.
    authorize(
        AgentRole.RESEARCH,
        "search_knowledge_base",
    )

    try:
        authorize(
            AgentRole.WRITER,
            "search_knowledge_base",
        )
    except PermissionDenied as exc:
        print(f"EXPECTED DENIAL: {exc}")
    else:
        raise AssertionError(
            "Unauthorized capability was not denied."
        )

    print("All permission tests passed.")


if __name__ == "__main__":
    main()