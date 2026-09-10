from typing import Any


class HybridRetriever:
    def __init__(
        self,
        vector_store,
        keyword_retriever,
    ):
        self.vector_store = vector_store
        self.keyword_retriever = keyword_retriever

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> list[dict[str, Any]]:

        vector_results = self.vector_store.search(
            query,
            top_k=candidate_k,
        )

        keyword_results = self.keyword_retriever.search(
            query,
            top_k=candidate_k,
        )

        scores = {}
        documents = {}

        # Vector ranking
        for rank, result in enumerate(vector_results, start=1):

            doc_id = result["id"]

            scores.setdefault(doc_id, 0.0)

            scores[doc_id] += 1.0 / (60 + rank)

            documents[doc_id] = result

        # Keyword ranking
        for rank, result in enumerate(keyword_results, start=1):

            doc_id = result["id"]

            scores.setdefault(doc_id, 0.0)

            scores[doc_id] += 1.0 / (60 + rank)

            documents.setdefault(doc_id, result)

        # Sort by fused score
        ranked_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )[:top_k]

        results = []

        for doc_id in ranked_ids:

            result = documents[doc_id].copy()

            result["hybrid_score"] = scores[doc_id]

            results.append(result)

        return results