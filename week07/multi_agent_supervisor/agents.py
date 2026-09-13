from typing import Any

import requests

from .state import MultiAgentState

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/chat"
TIMEOUT = 300


def ask_llm(prompt: str) -> str:
    """Call the local Ollama server through its HTTP API."""

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"].strip()


import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


WEEK06_ROOT = (
    Path(__file__).resolve().parents[2]
    / "week06"
    / "graph_mcp_agent"
)

MCP_SERVER = WEEK06_ROOT / "mcp_server.py"


async def call_mcp_search(query: str) -> str:
    """Call the Week 6 MCP knowledge-base server."""

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            result = await session.call_tool(
                "search_knowledge_base",
                arguments={
                    "query": query,
                },
            )

            if not result.content:
                return "No research results were returned."

            content = result.content[0]

            if hasattr(content, "text"):
                return content.text

            return str(content)


def search_knowledge_base(query: str) -> str:
    """Synchronous wrapper around the MCP search."""

    return asyncio.run(call_mcp_search(query))


def research_agent(state: MultiAgentState) -> dict[str, Any]:
    """
    Research Agent.

    Uses the Week 6 MCP server to access the Week 3 RAG system.
    """

    goal = state["goal"]

    print("\n[RESEARCH] Querying MCP knowledge base...")
    print(f"[RESEARCH] Query: {goal}")

    research = search_knowledge_base(goal)

    print("[RESEARCH] MCP search completed.")

    return {
        "research": research,
        "completed_agents": state["completed_agents"] + ["research"],
    }

def analyst_agent(state: MultiAgentState) -> dict[str, Any]:
    """Analyst Agent."""

    goal = state["goal"]
    research = state["research"]

    prompt = f"""
You are the Analyst Agent in a multi-agent system.

Your responsibility is to analyze the research produced by another
agent and derive useful conclusions.

User goal:
{goal}

Research:
{research}

Analyze the information.

Identify:
- important findings
- comparisons
- advantages and disadvantages
- technical trade-offs
- conclusions supported by the research

Do not write the final response to the user.
"""

    analysis = ask_llm(prompt)

    return {
        "analysis": analysis,
        "completed_agents": state["completed_agents"] + ["analyst"],
    }


def writer_agent(state: MultiAgentState) -> dict[str, Any]:
    """Writer Agent."""

    goal = state["goal"]
    research = state["research"]
    analysis = state["analysis"]

    prompt = f"""
You are the Writer Agent in a multi-agent system.

Your responsibility is to produce the final answer to the user.

User goal:
{goal}

Research:
{research}

Analysis:
{analysis}

Write a clear, technically accurate answer.

Use the research and analysis provided.
Do not invent unsupported facts.
Structure the answer clearly.
"""

    draft = ask_llm(prompt)

    return {
        "draft": draft,
        "completed_agents": state["completed_agents"] + ["writer"],
    }