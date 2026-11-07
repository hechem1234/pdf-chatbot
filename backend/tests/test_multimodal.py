import base64
import json
from io import BytesIO
from unittest.mock import patch

from PIL import Image
import pytest

import ocr_service
import vision_service


def png_bytes():
    image = Image.new("RGB", (240, 80), "white")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_ocr_uses_french_english_and_arabic_reader_and_preserves_paragraphs():
    fake_reader = type("Reader", (), {
        "readtext": lambda self, image, detail, paragraph: [
            "Bonjour le monde", "Hello world", "مرحبا بالعالم"
        ]
    })()
    with patch.object(ocr_service, "_reader", return_value=fake_reader):
        text = ocr_service.extract_image_text(png_bytes())
    assert "Bonjour le monde" in text
    assert "Hello world" in text
    assert "مرحبا بالعالم" in text
    assert "\n\n" in text


def test_ocr_rejects_corrupt_image():
    try:
        ocr_service.extract_image_text(b"not an image")
    except ValueError as error:
        assert "unreadable" in str(error)
    else:
        raise AssertionError("Expected unreadable image to fail")


def test_vision_sends_image_to_ollama_chat_api():
    captured = {}

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        if request.full_url.endswith("/api/tags"):
            return Response({"models": [{"name": vision_service.VISION_MODEL}]})
        captured["body"] = json.loads(request.data)
        return Response({"message": {"content": "A labeled bar chart."}})

    with patch.object(vision_service.urllib.request, "urlopen", side_effect=fake_urlopen):
        answer = vision_service.analyze_image(png_bytes(), "Explain the chart")
    assert answer == "A labeled bar chart."
    assert captured["url"].endswith("/api/chat")
    assert captured["body"]["messages"][0]["content"] == "Explain the chart"
    assert base64.b64decode(captured["body"]["messages"][0]["images"][0]).startswith(b"\x89PNG")


@pytest.mark.parametrize(("question", "description"), [
    ("Identify the visible objects.", "A red mug beside a book."),
    ("Explain this technical diagram.", "A sensor connects to a controller."),
    ("Interpret the chart.", "The plotted values rise over time."),
])
def test_vision_handles_object_diagram_and_chart_requests(question, description):
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/api/tags"):
            return Response({"models": [{"name": vision_service.VISION_MODEL}]})
        return Response({"message": {"content": description}})

    with patch.object(vision_service.urllib.request, "urlopen", side_effect=fake_urlopen):
        assert vision_service.analyze_image(png_bytes(), question) == description


def test_prompt_keeps_pdf_page_and_image_method_references():
    from generation import create_prompt

    prompt = create_prompt("What does the figure show?", [
        {"source": "report.pdf", "page": 3, "type": "pdf", "text": "PDF passage"},
        {"source": "figure.png", "page": None, "type": "image",
         "extraction_method": "vision", "text": "Visual description"},
    ])
    assert "report.pdf" in prompt and "Page : 3" in prompt
    assert "figure.png" in prompt and "Type : image" in prompt
    assert "vision" in prompt


def test_retrieval_preserves_image_and_pdf_metadata():
    import retrieval

    class Collection:
        def count(self):
            return 2

        def query(self, **kwargs):
            return {
                "documents": [["Extracted label", "PDF conclusion"]],
                "metadatas": [[
                    {"source": "chart.png", "document_id": "a" * 64,
                     "type": "image", "extraction_method": "ocr"},
                    {"source": "report.pdf", "page": 8, "document_id": "b" * 64,
                     "type": "pdf", "extraction_method": "pdf_text"},
                ]],
                "distances": [[0.1, 0.2]],
            }

    with patch.object(retrieval, "get_collection", return_value=Collection()), \
         patch.object(retrieval, "create_embeddings", return_value=[[0.0, 1.0]]):
        results = retrieval.retrieve("What is the conclusion?")
    assert results[0]["type"] == "image"
    assert results[0]["extraction_method"] == "ocr"
    assert results[1]["page"] == 8


def test_chat_returns_citations_for_image_and_pdf_sources():
    import generation

    docs = [
        {"source": "chart.png", "page": None, "document_id": "a" * 64,
         "type": "image", "extraction_method": "vision", "text": "A rising chart."},
        {"source": "report.pdf", "page": 8, "document_id": "b" * 64,
         "type": "pdf", "extraction_method": "pdf_text", "text": "Revenue increased."},
    ]
    fake_response = type("Response", (), {
        "message": type("Message", (), {"content": "The chart rises [Source 1]."})()
    })()
    with patch.object(generation, "retrieve", return_value=docs), \
         patch.object(generation.ollama, "chat", return_value=fake_response):
        answer, sources = generation.ask_rag("What is shown?")
    assert "[Source 1]" in answer
    assert sources[0]["file"] == "chart.png"
    assert sources[0]["page"] is None
    assert sources[0]["extraction_method"] == "vision"
    assert sources[1]["page"] == 8
