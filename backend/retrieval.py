from embeddings import create_embeddings
from vectorstore import get_collection

from config import TOP_K


def retrieve(question):

    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = create_embeddings(
        [question]
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved = []

    for i, document in enumerate(results["documents"][0]):

        retrieved.append({
            "text": document,
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i].get("page"),
            "document_id": results["metadatas"][0][i].get("document_id"),
            "type": results["metadatas"][0][i].get("type", "pdf"),
            "extraction_method": results["metadatas"][0][i].get("extraction_method", "pdf_text"),
            "distance": results["distances"][0][i]
        })

    return retrieved
