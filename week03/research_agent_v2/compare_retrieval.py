from ingestion.prepare import prepare_document
from retrieval.keyword_search import KeywordRetriever
from retrieval.vector_store import VectorStore


KNOWLEDGE_FILE = (
    "week03/research_agent_v2/knowledge/robotics.txt"
)


def print_results(title, results):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for rank, result in enumerate(results, start=1):

        print(f"\nRank {rank}")
        print(f"ID: {result['id']}")

        if "distance" in result:
            print(f"Vector distance: {result['distance']}")

        if "score" in result:
            print(f"BM25 score: {result['score']}")

        print(f"Text: {result['text'][:250]}")


def main():

    documents = prepare_document(KNOWLEDGE_FILE)

    keyword_retriever = KeywordRetriever(documents)
    vector_store = VectorStore()

    queries = [
        "How can a drone navigate without GPS?",
        "What is visual odometry?",
        "How can an autonomous aircraft estimate its movement?",
        "SLAM",
    ]

    for query in queries:

        print("\n\n")
        print("#" * 80)
        print(f"QUERY: {query}")
        print("#" * 80)

        vector_results = vector_store.search(
            query,
            top_k=3,
        )

        keyword_results = keyword_retriever.search(
            query,
            top_k=3,
        )

        print_results(
            "VECTOR SEARCH",
            vector_results,
        )

        print_results(
            "BM25 KEYWORD SEARCH",
            keyword_results,
        )


if __name__ == "__main__":
    main()