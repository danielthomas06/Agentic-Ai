import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP


# ============================================================
# WEEK 3 RAG IMPORTS
# ============================================================

WEEK3_ROOT = (
    Path(__file__).resolve().parents[2]
    / "week03"
    / "research_agent_v2"
)

sys.path.insert(0, str(WEEK3_ROOT))

from ingestion.prepare import prepare_document
from retrieval.keyword_search import KeywordRetriever
from retrieval.vector_store import VectorStore
from retrieval.hybrid_search import HybridRetriever
from retrieval.reranker import Reranker


# ============================================================
# CONFIGURATION
# ============================================================

KNOWLEDGE_FILE = (
    WEEK3_ROOT
    / "knowledge"
    / "robotics.txt"
)


# ============================================================
# MCP SERVER
# ============================================================

mcp = FastMCP(
    "Week3 Research Tools"
)


# ============================================================
# INITIALIZE RAG
# ============================================================

print(
    "[MCP] Loading Week 3 RAG pipeline...",
    file=sys.stderr,
)

documents = prepare_document(
    str(KNOWLEDGE_FILE)
)

keyword_retriever = KeywordRetriever(
    documents
)

vector_store = VectorStore()

hybrid_retriever = HybridRetriever(
    vector_store=vector_store,
    keyword_retriever=keyword_retriever,
)

reranker = Reranker()

print(
    "[MCP] RAG pipeline ready.",
    file=sys.stderr,
)


# ============================================================
# RESEARCH TOOL
# ============================================================

@mcp.tool()
def search_knowledge_base(
    query: str,
) -> str:
    """
    Search the Week 3 knowledge base using hybrid
    retrieval and cross-encoder reranking.
    """

    print(
        f"[MCP] Research query: {query}",
        file=sys.stderr,
    )

    # --------------------------------------------------------
    # Hybrid retrieval
    # --------------------------------------------------------

    candidates = hybrid_retriever.search(
        query,
        top_k=8,
        candidate_k=8,
    )

    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    results = reranker.rerank(
        query,
        candidates,
        top_k=4,
    )

    if not results:

        return (
            "No relevant information was found "
            "in the knowledge base."
        )

    # --------------------------------------------------------
    # Build source-aware result
    # --------------------------------------------------------

    output = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "unknown",
        )

        page = metadata.get(
            "page"
        )

        if page:
            citation = (
                f"{source}, page {page}"
            )
        else:
            citation = source

        output.append(
            f"[Result {index}]\n"
            f"[Source: {citation}]\n"
            f"{result['text']}"
        )

    return "\n\n".join(output)


# ============================================================
# COMPARISON TOOL
# ============================================================

@mcp.tool()
def compare_methods(
    method_a: str,
    method_b: str,
) -> str:
    """
    Retrieve information about two methods so that
    the agent can compare them.
    """

    query_a = (
        f"{method_a} GPS-denied "
        f"drone navigation"
    )

    query_b = (
        f"{method_b} GPS-denied "
        f"drone navigation"
    )

    result_a = search_knowledge_base(
        query_a
    )

    result_b = search_knowledge_base(
        query_b
    )

    return (
        f"=== {method_a} ===\n"
        f"{result_a}\n\n"
        f"=== {method_b} ===\n"
        f"{result_b}"
    )


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    mcp.run(
        transport="stdio"
    )