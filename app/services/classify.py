from typing import List, Optional

import torch
from app.domain.entities import FileEntity, ClassifyEntity
from app.infrastructure.ml.paddle_runner import PaddleOCRRunner
from app.infrastructure.ml.classifier.model import OCRclassifier


class ClassifyService:
    def __init__(self):
        self.classifier = OCRclassifier()
        self.paddle_ocr = PaddleOCRRunner()

    def classify(self, file_entity: List[FileEntity]) -> List[ClassifyEntity]:
        results = []
        for file in file_entity:
            ocr_result = self.paddle_ocr.predict(file.file)
            page_entity = self.paddle_ocr.process_result(
                ocr_result, file.width, file.height
            )
            classify_entity = self.classifier.predict(
                page_entity, file.references_info
            )
            results.append(classify_entity)
            
        # Clear GPU cache after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return results
        
    def batch_classify(self, file_entities: List[FileEntity], batch_size: Optional[int] = 4) -> List[ClassifyEntity]:
        """Process multiple files in optimized batches
        
        Args:
            file_entities: List of files to classify
            batch_size: Number of files to process in each batch (default 4)
            
        Returns:
            List of classification results for all files
        """
        all_results = []
        
        # Process in batches for better GPU utilization
        for i in range(0, len(file_entities), batch_size):
            batch = file_entities[i:i+batch_size]
            batch_results = []
            
            # Use GPU optimizations if available
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    for file in batch:
                        ocr_result = self.paddle_ocr.predict(file.file)
                        page_entity = self.paddle_ocr.process_result(
                            ocr_result, file.width, file.height
                        )
                        classify_entity = self.classifier.predict(
                            page_entity, file.references_info
                        )
                        batch_results.append(classify_entity)
                
                # Clear cache after each batch
                torch.cuda.empty_cache()
            else:
                # CPU processing
                for file in batch:
                    ocr_result = self.paddle_ocr.predict(file.file)
                    page_entity = self.paddle_ocr.process_result(
                        ocr_result, file.width, file.height
                    )
                    classify_entity = self.classifier.predict(
                        page_entity, file.references_info
                    )
                    batch_results.append(classify_entity)
            
            all_results.extend(batch_results)
            
        return all_results
