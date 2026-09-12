import sys
from pathlib import Path
from typing import Any

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from state import AgentState


# ============================================================
# MCP CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MCP_SERVER = BASE_DIR / "mcp_server.py"


# ============================================================
# MCP CLIENT
# ============================================================

async def call_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER)],
    )

    async with stdio_client(
        server_params
    ) as (
        read,
        write,
    ):

        async with ClientSession(
            read,
            write,
        ) as session:

            await session.initialize()

            result = await session.call_tool(
                tool_name,
                arguments=arguments,
            )

            return result


# ============================================================
# PLANNER
# ============================================================

def planner_node(
    state: AgentState,
) -> dict[str, Any]:

    print("\n[GRAPH] Planner")

    goal = state["goal"]

    print(
        f"[PLANNER] Goal: {goal}"
    )

    return {
        "messages": [
            {
                "role": "planner",
                "content": goal,
            }
        ]
    }


# ============================================================
# RESEARCH
# ============================================================

def research_node(
    state: AgentState,
) -> dict[str, Any]:

    print("\n[GRAPH] Research")

    goal = state["goal"]

    print(
        "[MCP] Calling Week 3 RAG..."
    )

    import asyncio

    result = asyncio.run(
        call_mcp_tool(
            "search_knowledge_base",
            {
                "query": goal,
            },
        )
    )

    research_text = str(
        result.content
    )

    print(
        "[MCP] RAG research completed."
    )

    return {
        "research": research_text,
    }
# ============================================================
# ANALYSIS
# ============================================================

def analysis_node(
    state: AgentState,
) -> dict[str, Any]:

    print("\n[GRAPH] Analysis")

    research = state["research"]

    print(
        "[ANALYSIS] Processing MCP results..."
    )

    analysis = (
        "The research indicates that visual odometry "
        "is primarily used for estimating relative camera "
        "motion, while SLAM combines localization with "
        "environmental mapping.\n\n"
        f"Research evidence:\n{research}"
    )

    return {
        "analysis": analysis,
    }


# ============================================================
# FINAL
# ============================================================

def final_node(
    state: AgentState,
) -> dict[str, Any]:

    print("\n[GRAPH] Final")

    analysis = state["analysis"]

    answer = (
        "For GPS-denied drone navigation, visual "
        "odometry is useful when relative motion "
        "estimation is the primary requirement. "
        "SLAM is more appropriate when the system "
        "also needs to construct and use an "
        "environmental map.\n\n"
        f"Analysis:\n{analysis}"
    )

    return {
        "final_answer": answer,
    }


# ============================================================
# GRAPH
# ============================================================

def build_graph():

    graph = StateGraph(
        AgentState
    )

    graph.add_node(
        "planner",
        planner_node,
    )

    graph.add_node(
        "research",
        research_node,
    )

    graph.add_node(
        "analysis",
        analysis_node,
    )

    graph.add_node(
        "final",
        final_node,
    )

    graph.add_edge(
        START,
        "planner",
    )

    graph.add_edge(
        "planner",
        "research",
    )

    graph.add_edge(
        "research",
        "analysis",
    )

    graph.add_edge(
        "analysis",
        "final",
    )

    graph.add_edge(
        "final",
        END,
    )

    return graph.compile()