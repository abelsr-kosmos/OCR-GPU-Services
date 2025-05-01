from io import BytesIO

import torch
import numpy as np
from PIL import Image

from app.infrastructure.ml.aligner.main import AlignerModel

class AlignService:
    def __init__(self) -> None:
        self._aligner = AlignerModel()

    def _preprocess_image(self, file: bytes) -> np.ndarray:
        """
        Convert image bytes to numpy array efficiently
        
        Args:
            file: bytes - Raw image bytes
            
        Returns:
            np.ndarray - Image as numpy array in RGB format
        """
        # Use memory-efficient loading
        img = Image.open(BytesIO(file))
        
        # Convert only if necessary - avoid unnecessary conversions
        if img.mode == 'RGB':
            return np.asarray(img)
        else:
            return np.asarray(img.convert("RGB"))

    def align(self, file: bytes) -> bytes:
        """
        Align an image from bytes
        
        Args:
            file: bytes - Raw image bytes
            
        Returns:
            bytes - Aligned image as JPEG bytes
        """
        img_array = self._preprocess_image(file)
        
        result = self._aligner.align_image(img_array)
                
        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result_bytes = BytesIO()
        result.save(result_bytes, format='JPEG', quality=95, optimize=True)
        result_bytes.seek(0)
        
        return result_bytes.getvalue()