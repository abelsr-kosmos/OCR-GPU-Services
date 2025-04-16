from fastapi import APIRouter, UploadFile, File, Query
from ...domain.entities.file_processing import ProcessingOptions
from ...application.services.file_processor import FileProcessorService
from ...infrastructure.repositories.file_processor_impl import FileProcessorRepositoryImpl

router = APIRouter(prefix="/api/v1", tags=["file-processing"])

# Initialize service with repository implementation
file_processor_service = FileProcessorService(FileProcessorRepositoryImpl())

@router.post("/classify-n-process-file")
async def classify_and_process_file(
    file: UploadFile = File(...),
    doctr: bool = Query(False, description="Enable docTR processing"),
    qr: bool = Query(False, description="Enable QR code detection"),
    signature: bool = Query(False, description="Enable signature detection")
):
    """Process and classify a file with optional docTR, QR, and signature detection."""
    options = ProcessingOptions(
        doctr=doctr,
        qr=qr,
        signature=signature
    )
    result = await file_processor_service.classify_and_process_file(
        file.file,
        options
    )
    return {
        "message": result.message,
        "status": result.status,
        "results": result.results
    }

@router.post("/process-file")
async def process_file(
    file: UploadFile = File(...),
    qr: bool = Query(False, description="Enable QR code detection"),
    signature: bool = Query(False, description="Enable signature detection")
):
    """Process a file with optional QR and signature detection."""
    options = ProcessingOptions(
        qr=qr,
        signature=signature
    )
    result = await file_processor_service.process_file(
        file.file,
        options
    )
    return {
        "message": result.message,
        "status": result.status,
        "results": result.results
    } 