import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


MCP_SERVER = (
    Path(__file__).resolve().parents[1]
    / "week06"
    / "graph_mcp_agent"
    / "mcp_server.py"
)


async def test_unauthorized_call() -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            print("MCP server initialized.")

            result = await session.call_tool(
                "search_knowledge_base",
                arguments={
                    "query": "visual odometry",
                    "agent_role": "writer",
                },
            )

            print(f"Tool error flag: {result.isError}")

            if result.isError:
                print(
                    "EXPECTED MCP DENIAL: "
                    "Unauthorized writer access was rejected."
                )

                for content in result.content:
                    if hasattr(content, "text"):
                        print(f"Server response: {content.text}")

                return

            raise AssertionError(
                "SECURITY FAILURE: "
                "Unauthorized writer access was allowed."
            )


if __name__ == "__main__":
    asyncio.run(test_unauthorized_call())