
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import hashlib

from ingestion import load_and_chunk_pdf
from embeddings import create_embeddings
from vectorstore import index_chunks
from generation import ask_rag
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from config import ALLOWED_IMAGE_EXTENSIONS, MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS
from image_service import ingest_image, list_images, delete_image, targeted_analysis

app = FastAPI(title="PDF RAG API")

# Autoriser Angular à communiquer avec FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ImageAnalysisRequest(BaseModel):
    question: str


@app.get("/api/health")
def health():
    return {"status": "online"}


@app.post("/api/documents/upload")
def upload_document(file: UploadFile = File(...)):

    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted."
        )

    pdf_bytes = file.file.read()

    if len(pdf_bytes) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="PDF too large. Maximum size is 15 MB."
        )

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty file."
        )

    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    try:
        chunks = load_and_chunk_pdf(
            pdf_bytes,
            filename
        )

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail=("No extractable text found in this PDF. It may be a scanned PDF; "
                        "OCR is not yet enabled for PDF pages.")
            )

        texts = [chunk["text"] for chunk in chunks]
        embeddings = create_embeddings(texts)

        count = index_chunks(
            chunks,
            embeddings,
            file_hash
        )

        return {
            "filename": file.filename,
            "chunks": count,
            "status": "indexed"
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not process PDF: {error}"
        )


@app.post("/api/chat")
def chat(request: ChatRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        answer, sources = ask_rag(request.question)

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.post("/api/images/upload")
async def upload_image(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Supported image formats are JPG, JPEG, PNG, and WEBP.")
    content = await file.read(MAX_IMAGE_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Empty image.")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the configured upload limit.")
    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions exceed the configured pixel limit.")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=422, detail="The uploaded image is unreadable or corrupted.")
    try:
        return ingest_image(filename, content)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail="Image processing failed.")


@app.get("/api/images")
def get_images():
    return {"images": list_images()}


@app.delete("/api/images/{image_id}")
def remove_image(image_id: str):
    if not delete_image(image_id):
        raise HTTPException(status_code=404, detail="Image not found.")
    return {"image_id": image_id, "status": "deleted"}


@app.post("/api/images/{image_id}/analyze")
def analyze_uploaded_image(image_id: str, request: ImageAnalysisRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        return {"image_id": image_id, "answer": targeted_analysis(image_id, request.question)}
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))
