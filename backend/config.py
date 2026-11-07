EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

LLM_MODEL = "qwen2.5:3b"

DATABASE_PATH = "./database"

COLLECTION_NAME = "pdf_documents"

CHUNK_SIZE = 500

CHUNK_OVERLAP = 100

TOP_K = 4

import os
from pathlib import Path

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:3b")
IMAGE_STORAGE_PATH = Path(os.getenv("IMAGE_STORAGE_PATH", "./image_storage"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "20000000"))
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
