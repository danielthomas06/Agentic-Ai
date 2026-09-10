from ingestion.prepare import prepare_document
from retrieval.vector_store import VectorStore


KNOWLEDGE_FILE = (
    "week03/research_agent_v2/knowledge/robotics.txt"
)


def main():

    documents = prepare_document(KNOWLEDGE_FILE)

    print(f"Prepared {len(documents)} chunks.")

    store = VectorStore()

    store.add_documents(documents)

    print("Documents indexed successfully.")


if __name__ == "__main__":
    main()