from typing import List

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
        return results
