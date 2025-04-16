from fastapi import APIRouter
from .routers.ocr import router as ocr_router
from .routers.classification import router as classification_router

# Create main v1 router
router = APIRouter(prefix="/v1")

# Include all routers
router.include_router(ocr_router)
router.include_router(classification_router) 