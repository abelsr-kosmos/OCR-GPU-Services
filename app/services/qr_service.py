from io import BytesIO
from typing import Tuple, Optional

import numpy as np
from PIL import Image
from qreader import QReader

class QRService:
    def __init__(self) -> None:
        self._reader = QReader()

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Converts image bytes to a NumPy array."""
        img = Image.open(BytesIO(image_bytes))
        return np.array(img)

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
        detections = self._reader.detect(image=np_image)
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
        # Call the detect_and_decode method of QReader
        # return_detections=False is default, explicitly stating for clarity if desired
        decoded_qrs = self._reader.detect_and_decode(image=np_image)
        return decoded_qrs