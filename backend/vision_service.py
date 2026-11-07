import base64
import json
import urllib.error
import urllib.request

from config import OLLAMA_HOST, VISION_MODEL


def _ensure_model_available() -> None:
    request = urllib.request.Request(f"{OLLAMA_HOST.rstrip('/')}/api/tags")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            models = json.loads(response.read().decode("utf-8")).get("models", [])
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not check available Ollama models at {OLLAMA_HOST}.") from exc
    names = {model.get("name") for model in models}
    if VISION_MODEL not in names:
        raise RuntimeError(
            f"Vision model '{VISION_MODEL}' is not available in Ollama. "
            f"Pull it with: ollama pull {VISION_MODEL}"
        )


def analyze_image(image_bytes: bytes, question: str | None = None) -> str:
    _ensure_model_available()
    prompt = question or (
        "Describe this image in a detailed, structured way. Identify visible objects, "
        "layout, diagrams, charts, labels, and relationships. Distinguish observations "
        "from uncertainty; do not guess unreadable text."
    )
    payload = json.dumps({
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": prompt,
                      "images": [base64.b64encode(image_bytes).decode("ascii")]}],
        "stream": False,
        "options": {"temperature": 0.1},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_HOST.rstrip('/')}/api/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama vision service at {OLLAMA_HOST}.") from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError("Ollama vision request timed out or returned invalid data.") from exc
    content = result.get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError(f"Vision model '{VISION_MODEL}' returned no description.")
    return content
