import time
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

import torch
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from fastapi.responses import ORJSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from loguru import logger

# Set CUDA optimizations for better performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.set_float32_matmul_precision('high')
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device for ML Inference: {'cuda' if torch.cuda.is_available() else 'cpu'}")


service_cache = {}
ml_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ml_worker")

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
        get_qr_service,
        get_paddle_service,
        get_doctr_service,
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
    
    # Log model summary
    try:
        summary = "\n--- Model Initialization Summary ---"
        for name, service in service_cache.items():
            device = "Unknown"
            # Attempt to get device info based on service type/attributes
            if hasattr(service, '_runner') and hasattr(service._runner, 'model') and hasattr(service._runner.model, 'device'): # DocTR
                device = str(service._runner.model.device)
            elif hasattr(service, '_paddle_runner') and hasattr(service._paddle_runner, 'use_gpu'): # Paddle
                device = "gpu" if service._paddle_runner.use_gpu else "cpu"
            elif hasattr(service, 'detector') and hasattr(service.detector, 'DEVICE'): # DocumentDetector service wrapper?
                device = service.detector.DEVICE
            elif hasattr(service, 'aligner') and hasattr(service.aligner, 'model') and hasattr(service.aligner.model, 'device'): # Align service wrapper?
                 device = str(service.aligner.model.device)
            elif hasattr(service, 'model') and hasattr(service.model, 'device'): # Generic check
                 device = str(service.model.device)
            elif hasattr(service, 'DEVICE'): # Direct attribute
                 device = service.DEVICE

            summary += f"\n- {name}: Initialized (Device: {device})"
        summary += "\n----------------------------------"
        logger.info(summary)
    except Exception as e:
        logger.warning(f"Could not generate model summary: {e}")
        
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
    """Explicitly trigger model downloads/loading using singleton service instances"""
    try:
        logger.info("Preloading models using singleton service instances...")
        # Import getter functions here to be used by _preload functions
        from app.dependencies import get_doctr_service, get_paddle_service

        await asyncio.gather(
            asyncio.to_thread(_preload_doctr_model, get_doctr_service),
            asyncio.to_thread(_preload_paddle_model, get_paddle_service)
        )
        logger.info("Model preloading completed successfully")
    except Exception as e:
        logger.error(f"Error during model preloading: {e}")

def _preload_doctr_model(getter_func):
    """Preloads DocTR model using its singleton service instance."""
    logger.info("Preloading DocTR model via singleton service...")
    doctr_service = getter_func() # Use the passed getter to get the singleton
    # Accessing the model attribute on the runner should trigger its initialization
    # if DocTRRunner's own singleton logic hasn't already done so.
    model = doctr_service._runner.model 
    logger.info(f"DocTR model preloaded: {type(model)}")
    return model

def _preload_paddle_model(getter_func):
    """Preloads PaddleOCR model using its singleton service instance."""
    logger.info("Preloading PaddleOCR model via singleton service...")
    paddle_service = getter_func() # Use the passed getter to get the singleton
    # Accessing the ocr attribute on the runner should trigger its initialization.
    # PaddleService.__init__ creates PaddleOCRRunner, which loads the model.
    ocr_engine = paddle_service._paddle_runner.ocr
    logger.info(f"PaddleOCR model preloaded: {type(ocr_engine)}")
    return ocr_engine