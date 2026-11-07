from vectorstore import get_collection
from sentence_transformers import SentenceTransformer

collection = get_collection()

model = SentenceTransformer("all-MiniLM-L6-v2")

question = "What are Mohamed's machine learning experiences?"

embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[embedding],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

for i, doc in enumerate(results["documents"][0]):
    print(f"\nResult {i+1}")
    print("Text:", doc)
    print("Metadata:", results["metadatas"][0][i])
    print("Distance:", results["distances"][0][i])