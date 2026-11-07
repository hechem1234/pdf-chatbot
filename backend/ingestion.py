from io import BytesIO
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_and_chunk_pdf(pdf_bytes, filename):

    reader = PdfReader(BytesIO(pdf_bytes))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if not text or not text.strip():
            continue

        parts = splitter.split_text(text)

        for part in parts:

            chunks.append({
                "text": part,
                "source": filename,
                "page": page_number
            })

    return chunks