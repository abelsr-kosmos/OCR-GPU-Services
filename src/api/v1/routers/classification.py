from fastapi import APIRouter, UploadFile, File, Query, Depends, BackgroundTasks
from typing import Optional
import asyncio
import torch

from ....config import setup_logging
from ....infrastructure.utils.image_processing import ImageProcessor
from ....infrastructure.services.ocr_service import OCRService
from ....infrastructure.services.classification_service import ClassificationService
from ....infrastructure.services.optional_services import OptionalServices
from ....ml import align_document, detect_document, detect_signature
from ..dependencies import get_ocr_service, get_classification_service, get_optional_services
from ..models import ProcessResponse
from ..utils import process_image, run_in_threadpool, run_concurrent_tasks

# Setup logging
logger = setup_logging()

# Create router
router = APIRouter(prefix="/classification", tags=["Classification"])

@router.post("/classify", response_model=ProcessResponse)
async def classify_and_process_file(
    file: UploadFile = File(...),
    doctr: bool = Query(False, description="Enable docTR processing"),
    qr: bool = Query(False, description="Enable QR code detection"),
    signature: bool = Query(False, description="Enable signature detection"),
    use_ml_alignment: bool = Query(False, description="Use ML-based alignment"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    ocr_service: OCRService = Depends(get_ocr_service),
    classification_service: ClassificationService = Depends(get_classification_service),
    optional_services: OptionalServices = Depends(get_optional_services)
):
    """
    Process and classify a file with optional docTR, QR, and signature detection.
    """
    try:
        logger.info(f"Processing file: {file.filename} with options - doctr: {doctr}, qr: {qr}, signature: {signature}, use_ml_alignment: {use_ml_alignment}")
        
        # Step 1: Read and process the image
        contents = await file.read()
        await file.close()
        image = await process_image(contents)
        
        # Step 2: Align the image
        if use_ml_alignment:
            logger.info("Using ML-based document alignment")
            aligned_image = await align_document(image)
        else:
            logger.info("Using basic document alignment")
            aligned_image, angle = await ImageProcessor.align_image(image)
        
        # Step 3: Run core tasks concurrently (detection and OCR)
        core_tasks = {
            "document_boxes": detect_document(aligned_image),
            "ocr_results": ocr_service.process_image(aligned_image)
        }
        
        core_results = await run_concurrent_tasks(core_tasks)
        document_boxes = core_results["document_boxes"]
        ocr_results = core_results["ocr_results"]
        
        logger.info(f"Document detection found {len(document_boxes)} documents")
        
        # Step 4: Classify the document (depends on OCR results)
        classification_results = await classification_service.classify_document(
            aligned_image, ocr_results
        )
        
        # Step 5: Set up optional processing tasks
        optional_tasks = {}
        
        if doctr:
            optional_tasks["doctr"] = optional_services.process_doctr(aligned_image)
            
        if qr:
            optional_tasks["qr_codes"] = optional_services.detect_qr(aligned_image)
            
        if signature:
            signature_tasks = {
                "ml_detection": detect_signature(aligned_image),
                "basic_detection": optional_services.detect_signature(aligned_image)
            }
            # Get signature detection results
            signature_results = await run_concurrent_tasks(signature_tasks)
            # Will add to results later
        
        # Execute optional tasks concurrently
        optional_results = await run_concurrent_tasks(optional_tasks)
        
        # Combine all results
        results = {
            "ocr_results": ocr_results,
            "classification": classification_results,
            "documents_detected": len(document_boxes),
            **optional_results
        }
        
        # Add signature results if we processed them
        if signature:
            results["signatures"] = signature_results
        
        # Clean up large objects
        def cleanup():
            del image
            del aligned_image
            del contents
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        return ProcessResponse(
            message=f"Error processing file: {str(e)}",
            status="error"
        )
