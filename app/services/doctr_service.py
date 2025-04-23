from app.infrastructure.ml.doctr_runner import DocTRRunner

class DocTRService:
    def __init__(self) -> None:
        self._runner = DocTRRunner()

    def render(self, file: bytes, file_type: str) -> str:
        return self._runner.render(
            file=file, 
            file_type=file_type
        )

    def ocr(self, file: bytes, file_type: str) -> str:
        return self._runner.ocr(
            file=file,
            file_type=file_type
        )