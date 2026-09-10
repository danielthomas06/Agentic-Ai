from ingestion.prepare import prepare_document
from retrieval.keyword_search import KeywordRetriever
from retrieval.vector_store import VectorStore
from retrieval.hybrid_search import HybridRetriever
from agent import ResearchAgent


KNOWLEDGE_FILES = [
    "week03/research_agent_v2/knowledge/robotics.txt",
]


def main():

    print("Loading knowledge base...")

    all_documents = []

    for path in KNOWLEDGE_FILES:

        documents = prepare_document(path)

        all_documents.extend(documents)

    print(
        f"Loaded {len(all_documents)} document chunks."
    )

    # Vector database
    vector_store = VectorStore()

    vector_store.add_documents(all_documents)

    # BM25
    keyword_retriever = KeywordRetriever(
        all_documents
    )

    # Hybrid retrieval
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        keyword_retriever=keyword_retriever,
    )

    # Agent
    agent = ResearchAgent(
        hybrid_retriever
    )

    print("\nResearch Agent v2")
    print("Type 'exit' to quit.\n")

    while True:

        query = input("You: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        try:

            answer = agent.answer(query)

            print("\nAgent:")
            print(answer)
            print()

        except Exception as exc:

            print(
                f"\nError: {exc}\n"
            )


if __name__ == "__main__":
    main()