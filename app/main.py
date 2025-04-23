import os
from pathlib import Path

# Set common model cache directories for all workers
# These must be set before importing any ML libraries
MODEL_CACHE_ROOT = str(Path("/app/model_cache").absolute())
os.environ["DOCTR_CACHE_DIR"] = str(Path(f"{MODEL_CACHE_ROOT}/doctr").absolute())
os.environ["PADDLE_CACHE_DIR"] = str(Path(f"{MODEL_CACHE_ROOT}/paddle").absolute())
os.environ["TORCH_HOME"] = str(Path(f"{MODEL_CACHE_ROOT}/torch").absolute())

import time
import asyncio

import torch
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from fastapi.responses import ORJSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.v1.tools import router as tools_router
from app.api.v1.detect import router as detect_router
from app.api.v1.classify import router as classify_router
from app.dependencies import (
    get_paddle_service,
    get_doctr_service,
    get_qr_service,
    get_signature_service,
    get_document_detection_service,
    get_classify_service
)
from app.services.doctr_service import DocTRService
from app.services.paddle_service import PaddleService

from loguru import logger

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
logger.info(f"Using device for ML Inference: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Dictionary to store service instances
service_cache = {}

class ExecutionTimeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log the execution time
        logger.info(f"Request to {request.url.path} took {process_time:.4f} seconds")
        
        return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize services only once at application startup
    logger.info("Initializing services...")
    
    # Pre-load all services during startup
    service_cache["paddle_service"] = get_paddle_service()
    service_cache["doctr_service"] = get_doctr_service()
    service_cache["qr_service"] = get_qr_service()
    service_cache["signature_service"] = get_signature_service()
    service_cache["document_detection_service"] = get_document_detection_service()
    service_cache["classify_service"] = get_classify_service()
    
    logger.info("Services initialized successfully")
    yield
    # Clean up resources at shutdown
    service_cache.clear()
    logger.info("Services shut down")

app = FastAPI(
    title="Kosmos - OCR General GPU Services",
    description="Kosmos is a service that provides OCR General GPU Services",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse
)

# Add the execution time middleware
app.add_middleware(ExecutionTimeMiddleware)

@app.get("/")
async def root():
    return {"message": "Welcome to Kosmos - OCR General GPU Services to check docs go to /docs"}


@app.get("/health")
async def health():
    return {"status": "ok"}

app.include_router(tools_router, prefix="/tools")
app.include_router(detect_router, prefix="/detect")
app.include_router(classify_router, prefix="/classify")

# Preload heavy ML services at startup to avoid per-request initialization
@app.on_event("startup")
async def preload_services_event():
    # Ensure each service is instantiated only once
    get_paddle_service()
    get_doctr_service()
    get_qr_service()
    get_signature_service()
    get_document_detection_service()
    get_classify_service()

    # Explicitly preload models to avoid multiple downloads with multiple workers
    logger.info("Preloading models to shared cache directories...")
    
    # Run in background to avoid blocking startup
    asyncio.create_task(preload_models())

async def preload_models():
    """Explicitly trigger model downloads to shared cache location before workers need them"""
    try:
        # Create dummy services that will trigger model downloads to cache dirs
        doctr_service = DocTRService()
        paddle_service = PaddleService()
        
        # Force model loading by accessing the model attributes
        _ = doctr_service._runner.model
        _ = paddle_service._paddle_runner.ocr
        
        logger.info("Model preloading completed successfully")
    except Exception as e:
        logger.error(f"Error during model preloading: {e}")