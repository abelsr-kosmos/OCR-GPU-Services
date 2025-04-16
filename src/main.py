from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
from contextlib import asynccontextmanager
import torch

from .config import setup_logging, APP_SETTINGS, CORS_SETTINGS
from .api.router import router as api_router

# Setup logging
logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up OCR Processing Service")
    
    # Configure PyTorch to use CUDA by default
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_device(device)
        logger.info(f"PyTorch using CUDA: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logger.warning("CUDA not available. PyTorch will use CPU.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down OCR Processing Service")
    # Clean up GPU resources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA memory cache cleared")

app = FastAPI(
    **APP_SETTINGS,
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    **CORS_SETTINGS
)

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.2f}s")
    return response

# Include the API router
app.include_router(api_router) 