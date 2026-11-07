from functools import lru_cache
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from config import MAX_IMAGE_PIXELS


@lru_cache(maxsize=1)
def _reader():
    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError("EasyOCR is not installed. Install backend requirements.") from exc
    return easyocr.Reader(["en", "fr", "ar"], gpu=False)


def extract_image_text(image_bytes: bytes) -> str:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions exceed the configured pixel limit.")
            rgb = image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded image is unreadable or corrupted.") from exc
    # EasyOCR accepts numpy arrays (or paths/encoded bytes), not PIL.Image
    # instances. Passing the PIL object can make every otherwise-valid upload fail.
    results = _reader().readtext(np.asarray(rgb), detail=0, paragraph=True)
    return "\n\n".join(line.strip() for line in results if line and line.strip())
