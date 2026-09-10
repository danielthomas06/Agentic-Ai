from ingestion.prepare import prepare_document
from retrieval.keyword_search import KeywordRetriever
from retrieval.vector_store import VectorStore
from retrieval.hybrid_search import HybridRetriever


KNOWLEDGE_FILE = (
    "week03/research_agent_v2/knowledge/robotics.txt"
)


def main():

    # Prepare documents for BM25
    documents = prepare_document(KNOWLEDGE_FILE)

    keyword_retriever = KeywordRetriever(documents)

    # Existing persistent ChromaDB
    vector_store = VectorStore()

    # Combine both
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        keyword_retriever=keyword_retriever,
    )

    queries = [
        "How can a drone navigate without GPS?",
        "What is visual odometry?",
        "How does optical flow help robotics?",
        "What is SLAM?",
    ]

    for query in queries:

        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        results = hybrid_retriever.search(
            query,
            top_k=3,
            candidate_k=5,
        )

        for rank, result in enumerate(results, start=1):

            print(f"\nResult {rank}")
            print(f"ID: {result['id']}")
            print(f"Hybrid score: {result['hybrid_score']:.5f}")
            print(f"Metadata: {result['metadata']}")
            print(f"Text: {result['text'][:300]}")


if __name__ == "__main__":
    main()