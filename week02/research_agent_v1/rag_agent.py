import requests

from config import (
    OLLAMA_URL,
    MODEL,
)

from semantic_retrieval import (
    load_chunks,
    create_embeddings,
    retrieve,
)


def call_llm(prompt: str):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


def answer_question(
    question,
    chunks,
    embeddings,
):

    results = retrieve(
        question,
        chunks,
        embeddings,
        top_k=3,
    )

    if not results:
        return "No relevant information found."

    context_parts = []

    for result in results:

        context_parts.append(
            f"Source: {result['source']}\n"
            f"Relevance: {result['score']:.4f}\n"
            f"Content: {result['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a research assistant.

Answer the question using ONLY the
provided evidence.

If the evidence does not contain enough
information, say so explicitly.

Do not invent facts.

Evidence:

{context}

Question:

{question}
"""

    return call_llm(prompt)


if __name__ == "__main__":

    print("Loading knowledge base...")

    chunks = load_chunks()

    print(
        f"Loaded {len(chunks)} chunks."
    )

    print("Creating embeddings...")

    embeddings = create_embeddings(
        chunks
    )

    print("Ready.")

    question = input(
        "\nQuestion: "
    )

    answer = answer_question(
        question,
        chunks,
        embeddings,
    )

    print("\nAnswer:")
    print(answer)