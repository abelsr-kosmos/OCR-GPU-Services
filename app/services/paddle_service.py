from typing import List, Union

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
    
    def batch_ocr(self, images: List[Image.Image]) -> List[List[dict]]:
        """Process multiple images with PaddleOCR in batch
        
        This is more efficient than processing images one by one
        """
        # Convert all PIL images to numpy arrays
        image_nps = [np.array(img) for img in images]
        results = []
        
        # Process in batches of 4 for better GPU utilization
        batch_size = 4
        for i in range(0, len(image_nps), batch_size):
            batch = image_nps[i:i+batch_size]
            batch_results = [self._paddle_runner.predict(img) for img in batch]
            results.extend(batch_results)
            
        return results
