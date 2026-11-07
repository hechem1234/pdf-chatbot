import chromadb
import hashlib

from functools import lru_cache

from config import DATABASE_PATH, COLLECTION_NAME


@lru_cache(maxsize=1)
def get_collection():

    client = chromadb.PersistentClient(
        path=DATABASE_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def index_chunks(chunks, embeddings, file_hash):

    collection = get_collection()

    ids = []

    for i, chunk in enumerate(chunks):

        unique_text = f"{file_hash}:{chunk.get('type', 'pdf')}:{chunk.get('page', '')}:{chunk.get('extraction_method', '')}:{i}:{chunk['text']}"

        chunk_id = hashlib.sha256(
            unique_text.encode("utf-8")
        ).hexdigest()

        ids.append(chunk_id)

    collection.upsert(
        ids=ids,

        embeddings=embeddings,

        documents=[
            chunk["text"] for chunk in chunks
        ],

        metadatas=[{k: v for k, v in {
            "source": chunk["source"],
            "page": chunk.get("page"),
            "document_id": chunk.get("document_id", file_hash),
            "type": chunk.get("type", "pdf"),
            "extraction_method": chunk.get("extraction_method", "pdf_text"),
        }.items() if v is not None} for chunk in chunks]
    )

    return len(chunks)
