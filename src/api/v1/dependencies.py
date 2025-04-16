"""Dependencies for API v1 routes"""
from functools import lru_cache
from ...infrastructure.services.ocr_service import OCRService
from ...infrastructure.services.classification_service import ClassificationService
from ...infrastructure.services.optional_services import OptionalServices

# Use lru_cache to create singleton instances
@lru_cache()
def get_ocr_service():
    """Get OCR service instance (singleton)"""
    return OCRService()

@lru_cache()
def get_classification_service():
    """Get classification service instance (singleton)"""
    return ClassificationService()

@lru_cache()
def get_optional_services():
    """Get optional services instance (singleton)"""
    return OptionalServices() 