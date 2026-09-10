from pathlib import Path

from ingestion.loader import load_document
from ingestion.chunker import chunk_text


def prepare_document(path: str) -> list[dict]:

    documents = load_document(path)

    results = []

    for document in documents:

        chunks = chunk_text(document["text"])

        for chunk_index, chunk in enumerate(chunks):

            source = document["metadata"]["source"]

            page = document["metadata"].get("page")

            if page is not None:
                document_id = (
                    f"{Path(source).stem}"
                    f"_p{page}"
                    f"_c{chunk_index}"
                )
            else:
                document_id = (
                    f"{Path(source).stem}"
                    f"_c{chunk_index}"
                )

            metadata = {
                **document["metadata"],
                "chunk_index": chunk_index,
            }

            results.append(
                {
                    "id": document_id,
                    "text": chunk,
                    "metadata": metadata,
                }
            )

    return results