import time
import traceback
from io import BytesIO
from typing import Literal

from PIL import Image
from loguru import logger
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Query

from app.services.qr_service import QRService
from app.services.doctr_service import DocTRService
from app.services.align_service import AlignService
from app.services.paddle_service import PaddleService
from app.services.signature_service import SignatureService
from app.dependencies import get_paddle_service, get_doctr_service, get_qr_service, get_signature_service, get_align_service


router = APIRouter(
    prefix="",
    tags=["tools"],
)

@router.post("/paddle-ocr")
async def paddle_ocr(
    file: UploadFile = File(..., media_type=["image/jpeg", "image/png", "image/jpg", "application/pdf"], description="The image to detect documents in"), 
    paddle_service: PaddleService = Depends(get_paddle_service)
):
    """
    Extract text from an image using PaddleOCR
    
    Args:
        file: The image file to process
        paddle_service: The service to use for OCR
        
    Returns:
        Dictionary of OCR results
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "application/pdf"]:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    
    try:
        t_total = time.time()
        logger.info(f"Processing file: {file.filename}")
        
        # Read file with timing
        t_read = time.time()
        file_content = await file.read()
        file.file.seek(0)  # Reset file pointer for next read
        logger.info(f"File read completed in {time.time() - t_read:.4f} seconds")
        
        # Open image with PIL and timing
        t_open = time.time()
        image = Image.open(file.file)
        logger.info(f"Image opened in {time.time() - t_open:.4f} seconds")
        
        # Run OCR with timing
        t_ocr = time.time()
        result = await run_in_threadpool(paddle_service.ocr, image)
        logger.info(f"OCR processing completed in {time.time() - t_ocr:.4f} seconds")
        
        logger.info(f"Total processing time for {file.filename}: {time.time() - t_total:.4f} seconds")
        return result
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/doctr")
async def doctr(
    file: UploadFile = File(..., media_type=["image/jpeg", "image/png", "image/jpg", "application/pdf"], description="The image to detect documents in"), 
    operation: Literal["ocr", "render"] = Query(..., description="The operation to perform"),
    doctr_service: DocTRService = Depends(get_doctr_service)
):
    """
    Process an image with DocTR for OCR or visualization
    
    Args:
        file: The image file to process
        operation: The operation to perform (ocr or render)
        doctr_service: The service to use for DocTR processing
        
    Returns:
        OCR results or rendered visualization
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "application/pdf"]:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    try:
        t_total = time.time()
        logger.info(f"Processing file: {file.filename}")
        
        # Read file with timing
        t_read = time.time()
        file_bytes = await file.read()
        logger.info(f"File read completed in {time.time() - t_read:.4f} seconds")
        
        # Process with doctr
        t_process = time.time()
        if operation == "ocr":
            result = await run_in_threadpool(doctr_service.ocr, file_bytes, file.filename.split(".")[-1])
            logger.info(f"DocTR OCR processing completed in {time.time() - t_process:.4f} seconds")
        elif operation == "render":
            result = await run_in_threadpool(doctr_service.render, file_bytes, file.filename.split(".")[-1])
            logger.info(f"DocTR render processing completed in {time.time() - t_process:.4f} seconds")
        else:
            raise HTTPException(status_code=400, detail=f"Invalid operation: {operation}")
            
        logger.info(f"Total DocTR processing time for {file.filename}: {time.time() - t_total:.4f} seconds")
        return result
    except Exception as e:
        logger.error(f"Error processing {file.filename} with DocTR: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/qr")
async def qr(
    file: UploadFile = File(..., media_type=["image/jpeg", "image/png", "image/jpg", "application/pdf"], description="The image to detect documents in"), 
    qr_service: QRService = Depends(get_qr_service)
):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "application/pdf"]:
        raise HTTPException(status_code=415, detail="Unsupported file type. Received file type: " + file.content_type)
    try:
        logger.info(f"Processing file: {file.filename}")
        file_bytes = await file.read()
        t0 = time.time()
        result = await run_in_threadpool(qr_service.detect_decode, file_bytes)
        logger.info(f"Processed file: {file.filename} in {time.time() - t0} seconds")
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/signature")
async def signature(
    file: UploadFile = File(..., media_type=["image/jpeg", "image/png", "image/jpg", "application/pdf"], description="The image to detect documents in"), 
    return_img: bool = Query(False, description="Whether to return the image with the signatures detected"),
    signature_service: SignatureService = Depends(get_signature_service)
):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "application/pdf"]:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    try:
        logger.info(f"Processing file: {file.filename}")
        t0 = time.time()
        file_bytes = await file.read()
        result = await run_in_threadpool(signature_service.detect, file_bytes, return_img)
        logger.info(f"Processed file: {file.filename} in {time.time() - t0} seconds")
        return {"signatures": result}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router.post("/align")
async def align(
    file: UploadFile = File(..., media_type=["image/jpeg", "image/png", "image/jpg", "application/pdf"], description="The image to align"),
    align_service: AlignService = Depends(get_align_service)
):
    """
    Align a document image to correct perspective and return the aligned image
    
    Args:
        file: File uploaded by the user
        align_service: Service to handle alignment
        
    Returns:
        StreamingResponse containing the aligned image as JPEG
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "application/pdf"]:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    try:
        t_start = time.time()
        file_bytes = await file.read()
        
        # Alignment processing
        t_align = time.time()
        result = await run_in_threadpool(align_service.align, file_bytes)
        t_align_end = time.time()
        logger.info(f"Time for alignment service: {t_align_end - t_align} seconds")
        
        response = StreamingResponse(BytesIO(result), media_type="image/jpeg")
        return response
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
