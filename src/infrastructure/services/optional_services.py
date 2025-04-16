import cv2
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import pyzbar.pyzbar as pyzbar
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

logger = logging.getLogger(__name__)

class OptionalServices:
    """Services for optional document processing features"""
    
    def __init__(self):
        # Initialize docTR model
        self.doctr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
        logger.info("Optional services initialized successfully")
    
    async def process_doctr(self, image: np.ndarray) -> Dict[str, Any]:
        """Process document using docTR"""
        try:
            # Convert numpy array to bytes
            _, img_encoded = cv2.imencode('.png', image)
            img_bytes = img_encoded.tobytes()
            
            # Create DocumentFile from bytes
            doc = DocumentFile.from_images([img_bytes])
            
            # Process with docTR
            result = self.doctr_model(doc)
            
            # Format results
            formatted_results = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            formatted_results.append({
                                "text": word.value,
                                "confidence": float(word.confidence),
                                "position": word.geometry
                            })
            
            logger.info(f"docTR processed successfully. Found {len(formatted_results)} elements")
            return {
                "status": "success",
                "results": formatted_results
            }
            
        except Exception as e:
            logger.error(f"Error in docTR processing: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def detect_qr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect QR codes in the image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect QR codes
            decoded_objects = pyzbar.decode(gray)
            
            # Format results
            qr_results = []
            for obj in decoded_objects:
                qr_results.append({
                    "data": obj.data.decode('utf-8'),
                    "type": obj.type,
                    "position": obj.polygon
                })
            
            logger.info(f"QR detection completed. Found {len(qr_results)} QR codes")
            return qr_results
            
        except Exception as e:
            logger.error(f"Error in QR detection: {str(e)}", exc_info=True)
            return []
    
    async def detect_signature(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect signatures in the image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours that might be signatures
            signature_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                
                # Signature-like characteristics
                if 0.2 < aspect_ratio < 5.0 and 100 < w < 500 and 50 < h < 200:
                    signature_contours.append({
                        "position": (x, y, w, h),
                        "area": cv2.contourArea(contour)
                    })
            
            logger.info(f"Signature detection completed. Found {len(signature_contours)} potential signatures")
            return {
                "status": "success",
                "signatures": signature_contours
            }
            
        except Exception as e:
            logger.error(f"Error in signature detection: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            } 