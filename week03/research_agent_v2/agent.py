import requests

from memory.conversation import ConversationMemory
from memory.long_term import LongTermMemory
from retrieval.hybrid_search import HybridRetriever
from retrieval.reranker import Reranker


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"


class ResearchAgent:

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
    ):

        self.hybrid_retriever = hybrid_retriever

        self.reranker = Reranker()

        self.conversation = ConversationMemory()

        self.long_term = LongTermMemory()

    def answer(self, query: str) -> str:

        self.conversation.add_user_message(query)

        # Retrieve candidates
        candidates = self.hybrid_retriever.search(
            query,
            top_k=10,
            candidate_k=10,
        )

        # Rerank
        results = self.reranker.rerank(
            query,
            candidates,
            top_k=5,
        )

        # Build grounded context
        context_parts = []

        for result in results:

            metadata = result["metadata"]

            source = metadata.get("source", "unknown")

            page = metadata.get("page")

            if page is not None:
                citation = f"{source}, page {page}"
            else:
                citation = source

            context_parts.append(
                f"[Source: {citation}]\n"
                f"{result['text']}"
            )

        context = "\n\n".join(context_parts)

        memories = self.long_term.get_all()

        memory_context = "\n".join(
            f"- {memory}"
            for memory in memories
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant. "
                    "Answer using the supplied context. "
                    "Do not invent facts. "
                    "If the context is insufficient, say so. "
                    "Cite sources using the provided source labels."
                ),
            }
        ]

        if memory_context:

            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Relevant long-term memory:\n"
                        + memory_context
                    ),
                }
            )

        messages.extend(
            self.conversation.get_messages()[-6:]
        )

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Research context:\n\n"
                    f"{context}\n\n"
                    f"Question:\n{query}"
                ),
            }
        )

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
            },
            timeout=120,
        )

        response.raise_for_status()

        answer = response.json()["message"]["content"]

        self.conversation.add_assistant_message(answer)

        return answer