import time
from typing import Literal

from PIL import Image
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from app.services.qr_service import QRService
from app.services.doctr_service import DocTRService
from app.services.paddle_service import PaddleService
from app.services.signature_service import SignatureService
from app.dependencies import get_paddle_service, get_doctr_service, get_qr_service, get_signature_service

import traceback
from loguru import logger

router = APIRouter(
    prefix="",
    tags=["tools"],
)

@router.post("/paddle-ocr")
async def paddle_ocr(
    file: UploadFile = File(..., media_type=["image/jpeg", "image/png", "image/jpg"], description="The image to detect documents in"), 
    paddle_service: PaddleService = Depends(get_paddle_service)
):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    try:
        logger.info(f"Processing file: {file.filename}")
        image = Image.open(file.file)
        t0 = time.time()
        result = await run_in_threadpool(paddle_service.ocr, image)
        logger.info(f"Processed file: {file.filename} in {time.time() - t0} seconds")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/doctr")
async def doctr(
    file: UploadFile = File(..., media_type=["image/jpeg", "image/png", "image/jpg"], description="The image to detect documents in"), 
    operation: Literal["ocr", "render"] = Query(..., description="The operation to perform"),
    doctr_service: DocTRService = Depends(get_doctr_service)
):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    try:
        logger.info(f"Processing file: {file.filename}")
        t0 = time.time()
        file_bytes = await file.read()
        if operation == "ocr":
            result = await run_in_threadpool(doctr_service.ocr, file_bytes, file.filename.split(".")[-1])
        elif operation == "render":
            result = await run_in_threadpool(doctr_service.render, file_bytes, file.filename.split(".")[-1])
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/qr")
async def qr(
    file: UploadFile = File(..., media_type=["image/jpeg", "image/png", "image/jpg"], description="The image to detect documents in"), 
    qr_service: QRService = Depends(get_qr_service)
):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=415, detail="Unsupported file type")
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
    file: UploadFile = File(..., media_type=["image/jpeg", "image/png", "image/jpg"], description="The image to detect documents in"), 
    return_img: bool = Query(False, description="Whether to return the image with the signatures detected"),
    signature_service: SignatureService = Depends(get_signature_service)
):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
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