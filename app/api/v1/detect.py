import time
import traceback
from loguru import logger

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from starlette.concurrency import run_in_threadpool

from app.dependencies import get_document_detection_service
from app.services.doc_detector import DocumentDetector

router = APIRouter(
    prefix="/doc-detection",
    tags=["doc-detection"],
)

@router.post("/")
async def detect(
    file: UploadFile = File(..., media_type=["image/jpeg", "image/png", "image/jpg"], description="The image to detect documents in"),
    hide_result: bool = False,
    document_detection_service: DocumentDetector = Depends(get_document_detection_service)
):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    try:
        logger.info(f"Detecting documents in file: {file.filename}")
        t0 = time.time()
        
        # Use the regular thread pool instead of the specialized ML executor
        # This avoids circular imports with main.py
        detected_docs = await run_in_threadpool(document_detection_service.detect_docs, file)
        
        logger.info(f"Detected {len(detected_docs)} documents in {time.time() - t0} seconds")
        return {"success": True, "result": detected_docs if not hide_result else len(detected_docs)}
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
