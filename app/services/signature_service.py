from io import BytesIO
from typing import Tuple, List, Union

import numpy as np
from PIL import Image

from app.infrastructure.ml.signature_detector.main import SignatureDetector

class SignatureService:
    def __init__(self) -> None:
        self._detector = SignatureDetector()

    def _preprocess_image(self, file: bytes) -> np.ndarray:
        img = Image.open(BytesIO(file))
        return np.array(img)

    def detect(self, file: bytes, return_img: bool) -> Union[List, int]:  
        if return_img:
            result = self._detector.extract_signatures(self._preprocess_image(file))
            return result
        
        result = self._detector.detect_signatures(self._preprocess_image(file))
        
        return result