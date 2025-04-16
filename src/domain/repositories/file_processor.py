from abc import ABC, abstractmethod
from typing import BinaryIO
from ..entities.file_processing import FileProcessingResult, ProcessingOptions

class FileProcessorRepository(ABC):
    """Abstract base class for file processing repositories"""
    
    @abstractmethod
    async def process_file(
        self,
        file: BinaryIO,
        options: ProcessingOptions
    ) -> FileProcessingResult:
        """Process a file with the given options"""
        pass
    
    @abstractmethod
    async def classify_and_process_file(
        self,
        file: BinaryIO,
        options: ProcessingOptions
    ) -> FileProcessingResult:
        """Classify and process a file with the given options"""
        pass 