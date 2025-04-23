from app.infrastructure.ml.document_detector.main import DocumentDetector as InfrastructureDocumentDetector

class DocumentDetector:
    def __init__(self) -> None:
        self._detector = InfrastructureDocumentDetector()

    def detect_docs(self, file: str) -> list:
        docs = self._detector.detect(file)
        docs = [doc.tolist() for doc in docs if doc is not None]
        return docs