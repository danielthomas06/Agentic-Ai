import re
from typing import Any

from rank_bm25 import BM25Okapi


class KeywordRetriever:
    def __init__(self, documents: list[dict[str, Any]]):

        self.documents = documents

        tokenized_documents = [
            self._tokenize(document["text"])
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []

        for index in ranked_indices:

            document = self.documents[index]

            results.append(
                {
                    "id": document["id"],
                    "text": document["text"],
                    "metadata": document["metadata"],
                    "score": float(scores[index]),
                }
            )

        return results