from enum import Enum
from .security_audit import audit_logger

class AgentRole(str, Enum):
    RESEARCH = "research"
    ANALYST = "analyst"
    WRITER = "writer"


# Least-privilege policy.
# Each agent gets only the capabilities it actually needs.
AGENT_PERMISSIONS: dict[AgentRole, set[str]] = {
    AgentRole.RESEARCH: {
        "search_knowledge_base",
    },
    AgentRole.ANALYST: {
        "compare_methods",
    },
    AgentRole.WRITER: set(),
}


class PermissionDenied(Exception):
    """Raised when an agent attempts an unauthorized capability."""


def is_allowed(
    agent: AgentRole | str,
    capability: str,
) -> bool:
    """
    Check whether an agent is allowed to use a capability.
    """

    try:
        role = (
            agent
            if isinstance(agent, AgentRole)
            else AgentRole(agent)
        )
    except ValueError:
        return False

    allowed_capabilities = AGENT_PERMISSIONS.get(role, set())

    return capability in allowed_capabilities


def authorize(
    agent: AgentRole | str,
    capability: str,
) -> None:
    """
    Enforce the agent capability policy.
    """

    agent_name = (
        agent.value
        if isinstance(agent, AgentRole)
        else str(agent)
    )

    if not is_allowed(agent, capability):

        audit_logger.record(
            event="authorization",
            role=agent_name,
            capability=capability,
            outcome="denied",
            reason="capability_not_allowed",
        )

        raise PermissionDenied(
            f"Agent '{agent_name}' is not authorized "
            f"to use capability '{capability}'."
        )

    audit_logger.record(
        event="authorization",
        role=agent_name,
        capability=capability,
        outcome="allowed",
    )