from fastapi import Depends
from functools import lru_cache
from app.services.qr_service import QRService
from app.services.classify import ClassifyService
from app.services.align_service import AlignService
from app.services.doctr_service import DocTRService
from app.services.paddle_service import PaddleService
from app.services.doc_detector import DocumentDetector
from app.services.signature_service import SignatureService
try:
    from app.main import service_cache
except ImportError:
    service_cache = {}

_doctr_service_singleton = DocTRService()

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

def get_align_service() -> AlignService:
    if "align_service" not in service_cache:
        service_cache["align_service"] = AlignService()
    return service_cache["align_service"]