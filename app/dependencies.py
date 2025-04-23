from fastapi import Depends
from app.services.qr_service import QRService
from app.services.classify import ClassifyService
from app.services.doctr_service import DocTRService
from app.services.paddle_service import PaddleService
from app.services.doc_detector import DocumentDetector
from app.services.signature_service import SignatureService
from functools import lru_cache

# Import service_cache from main (avoiding circular import)
try:
    from app.main import service_cache
except ImportError:
    # For initialization in main.py (first import)
    service_cache = {}

# Single DocTRService instance to avoid reinitialization per request
_doctr_service_singleton = DocTRService()

# Create service provider functions with dependency caching
# FastAPI will automatically cache these dependencies

def get_classify_service() -> ClassifyService:
    if "classify_service" not in service_cache:
        service_cache["classify_service"] = ClassifyService()
    return service_cache["classify_service"]

def get_doctr_service() -> DocTRService:
    return _doctr_service_singleton

def get_qr_service() -> QRService:
    if "qr_service" not in service_cache:
        service_cache["qr_service"] = QRService()
    return service_cache["qr_service"]

def get_signature_service() -> SignatureService:
    if "signature_service" not in service_cache:
        service_cache["signature_service"] = SignatureService()
    return service_cache["signature_service"]

def get_document_detection_service() -> DocumentDetector:
    if "document_detection_service" not in service_cache:
        service_cache["document_detection_service"] = DocumentDetector()
    return service_cache["document_detection_service"]

def get_paddle_service() -> PaddleService:
    if "paddle_service" not in service_cache:
        service_cache["paddle_service"] = PaddleService()
    return service_cache["paddle_service"]