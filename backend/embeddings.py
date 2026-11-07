from functools import lru_cache
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model():

    return SentenceTransformer(EMBEDDING_MODEL)


def create_embeddings(texts):

    model = get_embedding_model()

    embeddings = model.encode(texts)

    return embeddings.tolist()