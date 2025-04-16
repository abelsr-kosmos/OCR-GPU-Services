from paddleocr import PaddleOCR
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class OCRService:
    """Service for OCR operations using PaddleOCR"""
    
    def __init__(self):
        # Initialize PaddleOCR with English and Spanish languages
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', det_lang='ml', show_log=False, use_gpu=True)
        logger.info("PaddleOCR initialized successfully")
    
    async def process_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process an image and extract text using PaddleOCR.
        Returns a list of dictionaries containing text and its location.
        """
        try:
            # Perform OCR
            result = self.ocr.ocr(image, cls=True)
            
            # Format the results
            formatted_results = []
            for line in result[0]:
                text = line[1][0]
                confidence = line[1][1]
                position = line[0]
                
                formatted_results.append({
                    "text": text,
                    "confidence": float(confidence),
                    "position": position
                })
            
            logger.info(f"OCR processed successfully. Found {len(formatted_results)} text elements")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}", exc_info=True)
            return [] 