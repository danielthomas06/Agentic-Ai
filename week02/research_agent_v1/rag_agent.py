import requests

from config import (
    OLLAMA_URL,
    MODEL,
)

from retrieval import (
    load_documents,
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


def answer_question(question: str):

    documents = load_documents()

    results = retrieve(
        question,
        documents,
    )

    if not results:
        return "I couldn't find relevant information."

    context = "\n\n".join(
        result["text"]
        for result in results[:3]
    )

    prompt = f"""
Answer the question using ONLY the provided context.

If the answer is not contained in the context,
say that the information is not available.

Context:
{context}

Question:
{question}
"""

    return call_llm(prompt)


if __name__ == "__main__":

    question = input("Question: ")

    answer = answer_question(question)

    print("\nAnswer:")
    print(answer)