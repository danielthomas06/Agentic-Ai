from ingestion.prepare import prepare_document
from retrieval.keyword_search import KeywordRetriever
from retrieval.vector_store import VectorStore
from retrieval.hybrid_search import HybridRetriever
from retrieval.reranker import Reranker


KNOWLEDGE_FILE = (
    "week03/research_agent_v2/knowledge/robotics.txt"
)


def main():

    documents = prepare_document(KNOWLEDGE_FILE)

    keyword_retriever = KeywordRetriever(documents)
    vector_store = VectorStore()

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        keyword_retriever=keyword_retriever,
    )

    reranker = Reranker()

    query = (
        "How can a drone estimate its movement "
        "when GPS is unavailable?"
    )

    print("\n" + "=" * 80)
    print("HYBRID RESULTS")
    print("=" * 80)

    candidates = hybrid_retriever.search(
        query,
        top_k=10,
        candidate_k=10,
    )

    for rank, result in enumerate(candidates, start=1):

        print(f"\nRank {rank}")
        print(f"ID: {result['id']}")
        print(f"Hybrid score: {result['hybrid_score']:.5f}")
        print(f"Text: {result['text'][:250]}")

    print("\n" + "=" * 80)
    print("RERANKED RESULTS")
    print("=" * 80)

    results = reranker.rerank(
        query,
        candidates,
        top_k=5,
    )

    for rank, result in enumerate(results, start=1):

        print(f"\nRank {rank}")
        print(f"ID: {result['id']}")
        print(f"Rerank score: {result['rerank_score']:.5f}")
        print(f"Hybrid score: {result['hybrid_score']:.5f}")
        print(f"Text: {result['text'][:300]}")


if __name__ == "__main__":
    main()