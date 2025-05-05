from app.infrastructure.ml.doctr_runner import DocTRRunner
import time
from loguru import logger

class DocTRService:
    def __init__(self) -> None:
        t_start = time.time()
        self._runner = DocTRRunner()
        logger.info(f"DocTRService initialized in {time.time() - t_start:.4f} seconds")

    def render(self, file: bytes, file_type: str) -> str:
        """Render document with detailed timing and error handling"""
        t_start = time.time()
        try:
            result = self._runner.render(
                file=file, 
                file_type=file_type
            )
            logger.info(f"DocTR render service completed in {time.time() - t_start:.4f} seconds")
            return result
        except Exception as e:
            logger.error(f"DocTR render error: {str(e)}")
            raise

    def ocr(self, file: bytes, file_type: str) -> str:
        """OCR document with detailed timing and error handling"""
        t_start = time.time()
        try:
            result = self._runner.ocr(
                file=file,
                file_type=file_type
            )
            logger.info(f"DocTR OCR service completed in {time.time() - t_start:.4f} seconds")
            return result
        except Exception as e:
            logger.error(f"DocTR OCR error: {str(e)}")
            raise