from typing import Dict, Any, List
import numpy as np
import logging
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

class ClassificationService:
    """Service for document classification"""
    
    def __init__(self):
        # Initialize PaddleOCR for classification
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', det_lang='ml', show_log=False, use_gpu=True)
        logger.info("Classification service initialized successfully")
    
    async def classify_document(self, image: np.ndarray, ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Classify the document based on its content and structure.
        Returns classification results including document type and confidence.
        """
        try:
            # Extract key information from OCR results
            text_content = " ".join([result["text"] for result in ocr_results])
            
            # Basic classification based on keywords
            document_type = "unknown"
            confidence = 0.0
            
            # Define document type keywords
            document_keywords = {
                "invoice": ["invoice", "factura", "bill", "cuenta"],
                "receipt": ["receipt", "recibo", "ticket", "comprobante"],
                "contract": ["contract", "contrato", "agreement", "acuerdo"],
                "id": ["id", "identification", "identificación", "dni", "passport"]
            }
            
            # Classify based on keywords
            for doc_type, keywords in document_keywords.items():
                matches = sum(1 for keyword in keywords if keyword.lower() in text_content.lower())
                if matches > 0:
                    document_type = doc_type
                    confidence = matches / len(keywords)
                    break
            
            logger.info(f"Document classified as: {document_type} with confidence: {confidence}")
            
            return {
                "document_type": document_type,
                "confidence": confidence,
                "text_content": text_content
            }
            
        except Exception as e:
            logger.error(f"Error in document classification: {str(e)}", exc_info=True)
            return {
                "document_type": "unknown",
                "confidence": 0.0,
                "text_content": ""
            } 