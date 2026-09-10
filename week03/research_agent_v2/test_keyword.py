from ingestion.prepare import prepare_document
from retrieval.keyword_search import KeywordRetriever


KNOWLEDGE_FILE = (
    "week03/research_agent_v2/knowledge/robotics.txt"
)


def main():

    documents = prepare_document(KNOWLEDGE_FILE)

    retriever = KeywordRetriever(documents)

    queries = [
        "visual odometry",
        "GPS denied navigation",
        "SLAM",
        "optical flow",
    ]

    for query in queries:

        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        results = retriever.search(query, top_k=3)

        for i, result in enumerate(results, start=1):

            print(f"\nResult {i}")
            print(f"ID: {result['id']}")
            print(f"Score: {result['score']}")
            print(f"Metadata: {result['metadata']}")
            print(f"Text: {result['text'][:300]}")


if __name__ == "__main__":
    main()