import time
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

import torch
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from fastapi.responses import ORJSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.services.doctr_service import DocTRService
from app.services.paddle_service import PaddleService

from loguru import logger

# Set CUDA optimizations for better performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.set_float32_matmul_precision('high')  # For PyTorch 2.0+
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device for ML Inference: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Dictionary to store service instances
service_cache = {}

# Create a dedicated thread pool executor for ML tasks
# This will improve performance by reusing threads for ML operations
ml_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ml_worker")

# Define the decorator here, but don't rely on imports from routers
def run_in_ml_executor(func):
    """Decorator to run a function in the ML thread pool executor"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.get_event_loop().run_in_executor(
            ml_executor, 
            functools.partial(func, *args, **kwargs)
        )
    return wrapper

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
    logger.info("Initializing services...")
    
    # Import here to avoid circular imports
    from app.dependencies import (
        get_paddle_service,
        get_doctr_service,
        get_qr_service,
        get_signature_service,
        get_document_detection_service,
        get_classify_service,
        get_align_service
    )
    
    # Initialize all services in parallel for faster startup
    await asyncio.gather(
        asyncio.to_thread(lambda: service_cache.update({"paddle_service": get_paddle_service()})),
        asyncio.to_thread(lambda: service_cache.update({"doctr_service": get_doctr_service()})),
        asyncio.to_thread(lambda: service_cache.update({"qr_service": get_qr_service()})),
        asyncio.to_thread(lambda: service_cache.update({"signature_service": get_signature_service()})),
        asyncio.to_thread(lambda: service_cache.update({"document_detection_service": get_document_detection_service()})),
        asyncio.to_thread(lambda: service_cache.update({"classify_service": get_classify_service()})),
        asyncio.to_thread(lambda: service_cache.update({"align_service": get_align_service()}))
    )
    
    logger.info("Services initialized successfully")
    yield
    # Clean shutdown
    ml_executor.shutdown(wait=True)
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

# Import routers after defining utilities to avoid circular imports
from app.api.v1.tools import router as tools_router
from app.api.v1.detect import router as detect_router
from app.api.v1.classify import router as classify_router

app.include_router(tools_router, prefix="/tools")
app.include_router(detect_router, prefix="/detect")
app.include_router(classify_router, prefix="/classify")


@app.on_event("startup")
async def preload_services_event():
    # Import here to avoid circular imports
    from app.dependencies import (
        get_paddle_service,
        get_doctr_service,
        get_qr_service,
        get_signature_service,
        get_document_detection_service,
        get_classify_service
    )
    
    # Preload models explicitly to avoid cold start issues
    asyncio.create_task(preload_models())

async def preload_models():
    """Explicitly trigger model downloads to shared cache directories"""
    try:
        logger.info("Preloading models to shared cache directories...")
        await asyncio.gather(
            asyncio.to_thread(_preload_doctr_model),
            asyncio.to_thread(_preload_paddle_model)
        )
        logger.info("Model preloading completed successfully")
    except Exception as e:
        logger.error(f"Error during model preloading: {e}")

def _preload_doctr_model():
    doctr_service = DocTRService()
    return doctr_service._runner.model

def _preload_paddle_model():
    paddle_service = PaddleService()
    return paddle_service._paddle_runner.ocr