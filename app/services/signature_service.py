from io import BytesIO
from typing import Tuple, List, Union

import torch
import numpy as np
from PIL import Image

from app.infrastructure.ml.signature_detector.main import SignatureDetector

class SignatureService:
    def __init__(self) -> None:
        self._detector = SignatureDetector()

    def _preprocess_image(self, file: bytes) -> np.ndarray:
        img = Image.open(BytesIO(file))
        return np.array(img.convert("RGB"))

    def detect(self, file: bytes, return_img: bool) -> Union[List, int]:
        # Preprocess image once
        img_array = self._preprocess_image(file)
        
        # Use GPU with mixed precision if available
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                if return_img:
                    result = self._detector.extract_signatures(img_array)
                else:
                    result = self._detector.detect_signatures(img_array)
                
            # Clean up GPU memory
            torch.cuda.empty_cache()
        else:
            if return_img:
                result = self._detector.extract_signatures(img_array)
            else:
                result = self._detector.detect_signatures(img_array)
        
        return result