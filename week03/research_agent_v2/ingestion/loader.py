from pathlib import Path
from pypdf import PdfReader


def load_text_file(path: str) -> list[dict]:
    file_path = Path(path)

    text = file_path.read_text(encoding="utf-8")

    return [
        {
            "text": text,
            "metadata": {
                "source": file_path.name,
                "file_type": "txt",
            },
        }
    ]


def load_pdf_file(path: str) -> list[dict]:
    file_path = Path(path)

    reader = PdfReader(str(file_path))

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append(
            {
                "text": text,
                "metadata": {
                    "source": file_path.name,
                    "file_type": "pdf",
                    "page": page_number,
                },
            }
        )

    return documents


def load_document(path: str) -> list[dict]:

    suffix = Path(path).suffix.lower()

    if suffix == ".txt":
        return load_text_file(path)

    if suffix == ".pdf":
        return load_pdf_file(path)

    raise ValueError(
        f"Unsupported file type: {suffix}"
    )