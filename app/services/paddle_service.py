from typing import List

import numpy as np
from PIL import Image
from app.infrastructure.ml.paddle_runner import PaddleOCRRunner

class PaddleService:
    def __init__(self) -> None:
        self._paddle_runner = PaddleOCRRunner()

    def ocr(self, image: Image.Image) -> List[dict]:
        """Process an image with PaddleOCR"""
        # Convert PIL image to numpy array
        image_np = np.array(image)
        return self._paddle_runner.predict(image_np)
