from retrieval.vector_store import VectorStore


def main():

    store = VectorStore()

    queries = [
        "How can a drone navigate without GPS?",
        "What is SLAM?",
        "How does optical flow work?",
    ]

    for query in queries:

        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        results = store.search(query, top_k=3)

        for i, result in enumerate(results, start=1):

            print(f"\nResult {i}")
            print(f"ID: {result['id']}")
            print(f"Distance: {result['distance']}")
            print(f"Metadata: {result['metadata']}")
            print(f"Text: {result['text'][:300]}")


if __name__ == "__main__":
    main()