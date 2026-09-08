from pathlib import Path


KNOWLEDGE_DIR = Path("knowledge")


def load_documents():
    documents = []

    for path in KNOWLEDGE_DIR.glob("*.txt"):
        text = path.read_text(encoding="utf-8")

        documents.append({
            "source": str(path),
            "text": text,
        })

    return documents


def retrieve(query: str, documents: list[dict]):
    query_words = set(
        query.lower().split()
    )

    results = []

    for document in documents:

        text = document["text"].lower()

        score = sum(
            1
            for word in query_words
            if word in text
        )

        if score > 0:
            results.append({
                "source": document["source"],
                "text": document["text"],
                "score": score,
            })

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return results


if __name__ == "__main__":

    documents = load_documents()

    results = retrieve(
        "GPS denied visual navigation",
        documents,
    )

    for result in results:
        print(result)