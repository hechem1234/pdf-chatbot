import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import IMAGE_STORAGE_PATH, MAX_IMAGE_BYTES
from embeddings import create_embeddings
from ocr_service import extract_image_text
from vectorstore import index_chunks
from vision_service import analyze_image


def _safe_name(filename: str) -> str:
    name = Path(filename or "image").name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._")
    return name[:120] or "image"


def _metadata_path(image_id: str) -> Path:
    if len(image_id) != 64 or any(char not in "0123456789abcdef" for char in image_id.lower()):
        raise ValueError("Invalid image ID.")
    return IMAGE_STORAGE_PATH / f"{image_id}.json"


def ingest_image(filename: str, content: bytes) -> dict:
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit.")
    file_hash = hashlib.sha256(content).hexdigest()
    ext = Path(filename).suffix.lower()
    IMAGE_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    manifest = _metadata_path(file_hash)
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8"))
    image_path = IMAGE_STORAGE_PATH / f"{file_hash}{ext}"
    image_path.write_bytes(content)
    safe_filename = _safe_name(filename)
    try:
        ocr_text = extract_image_text(content)
        ocr_status = "ready" if ocr_text.strip() else "no_text"
    except Exception as exc:
        # Keep image ingestion usable when OCR setup is unavailable; the vision
        # model can still describe and index the image.
        ocr_text = ""
        ocr_status = "unavailable"
        print(f"OCR indexing failed for {safe_filename}: {exc}")
    try:
        visual_text = analyze_image(content)
        vision_status = "ready"
    except Exception as exc:
        visual_text = ""
        vision_status = "unavailable"
        print(f"Vision indexing failed for {safe_filename}: {exc}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = []
    for method, text in (("ocr", ocr_text), ("vision", visual_text)):
        if not text.strip():
            continue
        for part in splitter.split_text(text):
            chunks.append({
                "text": part,
                "source": safe_filename,
                "document_id": file_hash,
                "type": "image",
                "extraction_method": method,
            })
    if not chunks:
        image_path.unlink(missing_ok=True)
        raise RuntimeError("Could not read this image: OCR found no text and vision analysis is unavailable.")
    index_chunks(chunks, create_embeddings([item["text"] for item in chunks]), file_hash)
    record = {
        "image_id": file_hash,
        "document_id": file_hash,
        "filename": safe_filename,
        "size": len(content),
        "chunks": len(chunks),
        "ocr_status": ocr_status,
        "vision_status": vision_status,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return record


def list_images() -> list[dict]:
    if not IMAGE_STORAGE_PATH.exists():
        return []
    records = []
    for manifest in IMAGE_STORAGE_PATH.glob("*.json"):
        try:
            records.append(json.loads(manifest.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)


def delete_image(image_id: str) -> bool:
    try:
        manifest = _metadata_path(image_id)
    except ValueError:
        return False
    if not manifest.exists():
        return False
    record = json.loads(manifest.read_text(encoding="utf-8"))
    from vectorstore import get_collection
    get_collection().delete(where={"document_id": record["document_id"]})
    for path in IMAGE_STORAGE_PATH.glob(f"{image_id}.*"):
        if path.is_file():
            path.unlink(missing_ok=True)
    return True


def targeted_analysis(image_id: str, question: str) -> str:
    try:
        record = _metadata_path(image_id)
    except ValueError as exc:
        raise FileNotFoundError("Image not found.") from exc
    if not record.exists():
        raise FileNotFoundError("Image not found.")
    image = next(IMAGE_STORAGE_PATH.glob(f"{image_id}.*"), None)
    if image is None or image.suffix == ".json":
        raise FileNotFoundError("Original image is unavailable.")
    return analyze_image(image.read_bytes(), question)
