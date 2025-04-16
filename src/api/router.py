from fastapi import APIRouter
from .v1.router import router as v1_router

# Create main API router
router = APIRouter(prefix="/api")

# Include all version routers
router.include_router(v1_router) 