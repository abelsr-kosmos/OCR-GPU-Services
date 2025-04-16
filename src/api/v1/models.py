"""Common API models for v1 routes"""
from typing import Optional
from pydantic import BaseModel

class ProcessResponse(BaseModel):
    """Standard response model for processing endpoints"""
    message: str
    status: str
    results: Optional[dict] = None 