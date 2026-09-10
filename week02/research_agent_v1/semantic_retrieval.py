from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


KNOWLEDGE_DIR = Path("knowledge")

MODEL_NAME = "all-MiniLM-L6-v2"


# ------------------------------------------------------------
# Load embedding model
# ------------------------------------------------------------

model = SentenceTransformer(MODEL_NAME)


# ------------------------------------------------------------
# Load documents and create chunks
# ------------------------------------------------------------

def load_chunks():

    chunks = []

    for path in KNOWLEDGE_DIR.glob("*.txt"):

        text = path.read_text(
            encoding="utf-8"
        )

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        for paragraph in paragraphs:

            chunks.append({
                "source": str(path),
                "text": paragraph,
            })

    return chunks


# ------------------------------------------------------------
# Create embeddings
# ------------------------------------------------------------

def create_embeddings(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )

    return np.asarray(embeddings)


# ------------------------------------------------------------
# Semantic retrieval
# ------------------------------------------------------------

def retrieve(
    query: str,
    chunks,
    embeddings,
    top_k=3,
):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    scores = embeddings @ query_embedding

    indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in indices:

        results.append({
            "source": chunks[index]["source"],
            "text": chunks[index]["text"],
            "score": float(scores[index]),
        })

    return results


# ------------------------------------------------------------
# Test
# ------------------------------------------------------------

if __name__ == "__main__":

    chunks = load_chunks()

    embeddings = create_embeddings(
        chunks
    )

    query = input("Query: ")

    results = retrieve(
        query,
        chunks,
        embeddings,
        top_k=3,
    )

    print("\nResults:")

    for result in results:

        print(
            f"\nScore: {result['score']:.4f}"
        )

        print(
            f"Source: {result['source']}"
        )

        print(
            result["text"]
        )