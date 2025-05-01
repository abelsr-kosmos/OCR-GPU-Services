from io import BytesIO
from typing import Tuple, Optional, List

import cv2
import torch
import numpy as np
from PIL import Image
from qreader import QReader

from loguru import logger

class QRService:
    def __init__(self) -> None:
        self._reader = QReader()

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Converts image bytes to a NumPy array with preprocessing for better QR detection."""
        img = Image.open(BytesIO(image_bytes))
        
        # Convert to numpy array
        np_img = np.array(img)
        
        # Resize large images to reduce processing time
        # Only resize if larger than 1500 pixels in any dimension
        height, width = np_img.shape[:2]
        max_dim = 1500
        if height > max_dim or width > max_dim:
            scale = max_dim / max(height, width)
            new_height, new_width = int(height * scale), int(width * scale)
            np_img = cv2.resize(np_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
        return np_img

    def detect(self, file: bytes) -> Tuple[np.ndarray, ...]:
        """
        Detects QR codes in the image.

        Args:
            file: The image file content as bytes.

        Returns:
            A tuple containing detected bounding boxes.
            Refer to QReader documentation for the exact structure.
        """
        np_image = self._preprocess_image(file)
        with torch.cuda.amp.autocast():
            detections = self._reader.detect(image=np_image)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return detections

    def detect_decode(self, file: bytes) -> Tuple[Optional[str], ...]:
        """
        Detects and decodes QR codes in the image.

        Args:
            file: The image file content as bytes.

        Returns:
            A tuple containing the decoded content of the QR codes found.
            Returns None for codes that couldn't be decoded.
            Refer to QReader documentation for the exact structure.
        """
        np_image = self._preprocess_image(file)
        with torch.cuda.amp.autocast():
            # Detect and decode QR codes
            decoded_qrs = self._reader.detect_and_decode(image=np_image, return_detections=True)

        results = []
        for i in range(len(decoded_qrs[0])):
            results.append(
                {
                    "text": decoded_qrs[0][i],
                    "bbox_xyxy": decoded_qrs[1][i]['bbox_xyxy'].tolist(),
                }
            )
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return results
        
    def batch_detect_decode(self, files: List[bytes]) -> List[Tuple[Optional[str], ...]]:
        """
        Detects and decodes QR codes in multiple images.
        
        Args:
            files: List of image file contents as bytes.
            
        Returns:
            List of tuples containing decoded QR codes for each image.
        """
        results = []
        for file in files:
            np_image = self._preprocess_image(file)
            decoded_qrs = self._reader.detect_and_decode(image=np_image, return_detections=False)
            results.append(decoded_qrs)
        return results