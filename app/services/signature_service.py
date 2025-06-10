from io import BytesIO
from typing import Tuple, List, Union

import torch
import numpy as np
from PIL import Image
from fastapi import HTTPException
from pdf2image import convert_from_bytes

from app.infrastructure.ml.signature_detector.main import SignatureDetector

class SignatureService:
    def __init__(self) -> None:
        self._detector = SignatureDetector()

    def _preprocess_image(self, file: bytes) -> np.ndarray:
        try:
            img = Image.open(BytesIO(file))
            return [np.array(img.convert("RGB"))]
        except Exception as e:
            try:
                images = convert_from_bytes(file)
                return images
            except Exception as e:
                raise HTTPException(status_code=415, detail="Unsupported file type. Exception: " + str(e))

    def detect(self, file: bytes, return_img: bool) -> Union[List, int]:
        # Preprocess image once
        imgs_array = self._preprocess_image(file)
        
        results = []
        for img_array in imgs_array:
            if return_img:
                result = self._detector.extract_signatures(img_array)
            else:
                result = self._detector.detect_signatures(img_array)
            results.append(result)
                
            # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result