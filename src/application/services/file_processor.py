from typing import BinaryIO
from ..domain.entities.file_processing import FileProcessingResult, ProcessingOptions
from ..domain.repositories.file_processor import FileProcessorRepository

class FileProcessorService:
    """Service layer for file processing operations"""
    
    def __init__(self, repository: FileProcessorRepository):
        self.repository = repository
    
    async def process_file(
        self,
        file: BinaryIO,
        options: ProcessingOptions
    ) -> FileProcessingResult:
        """Process a file with the given options"""
        return await self.repository.process_file(file, options)
    
    async def classify_and_process_file(
        self,
        file: BinaryIO,
        options: ProcessingOptions
    ) -> FileProcessingResult:
        """Classify and process a file with the given options"""
        return await self.repository.classify_and_process_file(file, options) 