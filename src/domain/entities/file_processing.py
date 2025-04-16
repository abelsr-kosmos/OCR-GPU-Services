from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class FileProcessingResult:
    """Entity representing the result of file processing"""
    status: str
    message: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class ProcessingOptions:
    """Entity representing processing options"""
    doctr: bool = False
    qr: bool = False
    signature: bool = False 