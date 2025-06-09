from fastapi import APIRouter, UploadFile, File, Query, Depends, BackgroundTasks, HTTPException, status
from typing import Optional, List
import asyncio
import cv2
import torch

from ....config import setup_logging
from ....infrastructure.utils.image_processing import ImageProcessor
from ....infrastructure.services.ocr_service import OCRService
from ....infrastructure.services.optional_services import OptionalServices
from ..dependencies import get_ocr_service, get_optional_services
from ..models import ProcessResponse
from ..utils import process_image, run_in_threadpool, run_concurrent_tasks

# Import docTR components
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Setup logging
logger = setup_logging()

# Create router
router = APIRouter(prefix="/ocr", tags=["OCR"])

# Initialize docTR model with GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
logger.info(f"DocTR using device: {device}")
model = ocr_predictor(pretrained=True, det_arch='db_resnet50', reco_arch='crnn_vgg16_bn')

@router.post("/process-file", response_model=ProcessResponse)
async def process_file(
    file: UploadFile = File(...),
    qr: bool = Query(False, description="Enable QR code detection"),
    signature: bool = Query(False, description="Enable signature detection"),
    return_signatures: bool = Query(False, description="Return signatures in the response"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    optional_services: OptionalServices = Depends(get_optional_services)
):
    """
    Process a file with optional QR and signature detection. OCR is using the docTR package.
    Returns only the rendered text from docTR. Processing runs on GPU for faster performance.
    """
    try:
        logger.info(f"Processing file: {file.filename} with options - qr: {qr}, signature: {signature}")
        
        # Step 1: Read and process the image
        contents = await file.read()
        await file.close()
        
        # Check file extension
        extension = file.filename.split(".")[-1].lower()
        if extension not in ['jpg', 'jpeg', 'png', 'pdf']:
            logger.error(f"Unsupported file format: {extension}")
            raise HTTPException(
                status_code=400,
                detail=f"File format '{extension}' not supported. Use jpg, jpeg, png, or pdf."
            )
        
        # Step 2: Process file based on type
        if extension in ['jpg', 'jpeg', 'png']:
            # Create DocumentFile from image
            doc = DocumentFile.from_images([contents])
        else:  # PDF file
            doc = DocumentFile.from_pdf(contents)
            # For optional processing, we'll use the first page
            image = await process_image(doc.get_page_image(0))
            aligned_image = image  # No alignment for PDF pages

        # Step 3: Perform docTR OCR (run in threadpool since it's CPU/GPU intensive)
        result = await run_in_threadpool(lambda: model(doc))
        
        # Get the rendered text (plain text extraction)
        extracted_text = result.render()
        logger.info(f"DocTR OCR completed: extracted {len(extracted_text)} characters of text")
        
        # Step 4: Run optional processing tasks concurrently
        optional_tasks = {}
        
        if qr:
            # For images, we already have the image loaded
            if extension in ['jpg', 'jpeg', 'png']:
                image = await process_image(contents)
            
            optional_tasks["qr_codes"] = optional_services.detect_qr(image)
            
        if signature:
            # For images, use the same image as QR detection
            if extension in ['jpg', 'jpeg', 'png'] and 'image' not in locals():
                image = await process_image(contents)
            
            optional_tasks["signatures"] = optional_services.detect_signature(image)
        
        # Execute optional tasks concurrently
        optional_results = await run_concurrent_tasks(optional_tasks)
        
        # Combine all results - only include the rendered text for OCR
        results = {
            "text": extracted_text,
            **optional_results
        }

        if signature:
            if return_signatures:
                results["signatures"] = optional_results["signatures"]
            else:
                results["signatures"] = len(optional_results["signatures"]) 
        
        # Clean up large objects
        def cleanup():
            if 'image' in locals():
                del image
            if 'doc' in locals():
                del doc
            del contents
            # Explicitly clear CUDA cache to free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            
        background_tasks.add_task(cleanup)
        
        return ProcessResponse(
            message="File processed successfully",
            status="success",
            results=results
        )
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        # Ensure GPU memory is cleared even on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        return ProcessResponse(
            message=f"Error processing file: {str(e)}",
            status="error"
        )
