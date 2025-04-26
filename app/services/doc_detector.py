from PIL import Image
import torch
from app.infrastructure.ml.document_detector.main import DocumentDetector as InfrastructureDocumentDetector

class DocumentDetector:
    def __init__(self) -> None:
        self._detector = InfrastructureDocumentDetector()

    def detect_docs(self, file) -> list:
        # Read file content
        content = file.file.read()
        
        # Use CUDA if available
        if torch.cuda.is_available():
            # Ensure we're using the GPU
            with torch.cuda.amp.autocast():
                docs = self._detector.detect(content)
        else:
            docs = self._detector.detect(content)
        
        # Convert results to list format
        docs = [doc.tolist() for doc in docs if doc is not None]
        
        # Clear CUDA cache to avoid memory leaks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return docs