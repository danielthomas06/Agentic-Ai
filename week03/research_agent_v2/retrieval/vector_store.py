import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(
        self,
        persist_directory: str = "week03/research_agent_v2/chroma_db",
        collection_name: str = "research_documents",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def add_documents(self, documents: list[dict]) -> None:
        texts = [doc["text"] for doc in documents]
        ids = [doc["id"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True
        ).tolist()

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        output = []

        for doc_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            output.append(
                {
                    "id": doc_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return output