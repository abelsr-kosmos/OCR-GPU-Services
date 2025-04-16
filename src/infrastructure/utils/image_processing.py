import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Utility class for image processing operations"""
    
    @staticmethod
    async def align_image(image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Align the image using edge detection and perspective transformation.
        Returns the aligned image and the rotation angle.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour (assuming it's the document)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get the minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[-1]
            
            # Rotate the image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            logger.info(f"Image aligned with rotation angle: {angle} degrees")
            return rotated, angle
            
        except Exception as e:
            logger.error(f"Error in image alignment: {str(e)}", exc_info=True)
            return image, 0.0 