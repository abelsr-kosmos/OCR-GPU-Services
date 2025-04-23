from typing import Literal, Any

import torch
import os
from pathlib import Path
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from app.schemas.schemas import OCROut, OCRPage, OCRBlock, OCRLine, OCRWord

import time
from loguru import logger

# Get cache directory from environment or use default
DOCTR_CACHE_DIR = os.environ.get("DOCTR_CACHE_DIR", str(Path("./model_cache/doctr").absolute()))
os.makedirs(DOCTR_CACHE_DIR, exist_ok=True)
logger.info(f"Using DocTR cache directory: {DOCTR_CACHE_DIR}")

class DocTRRunner:
    def __init__(self):
        # Use cache_dir parameter to control where pretrained models are downloaded
        self.model = ocr_predictor(pretrained=True, cache_dir=DOCTR_CACHE_DIR)
        if torch.cuda.is_available():
            self.model.to("cuda").eval()
            logger.info("Using GPU for OCR")
        else:
            self.model.to("cpu").eval()
            logger.info("Using CPU for OCR")

    def render(
        self, file: bytes, file_type: Literal["image", "pdf"]
    ) -> list[dict]:
        start_time = time.time()
        if file_type in ["jpg", "jpeg", "png"]:
            doc = DocumentFile.from_images([file])
        elif file_type == "pdf":
            doc = DocumentFile.from_pdf(file)
        else:
            raise ValueError("Invalid file type")

        try:
            start_time = time.time()
            result = self.model(doc).render()
            end_time = time.time()
            logger.info(f"Time taken to render document: {end_time - start_time} seconds")
            return result
        except Exception as e:
            raise ValueError(f"Error rendering document: {e}")

    def ocr(
        self, file: bytes, file_type: Literal["image", "pdf"]
    ) -> list[dict]:
        if file_type in ["jpg", "jpeg", "png"]:
            doc = DocumentFile.from_images([file])
        elif file_type == "pdf":
            doc = DocumentFile.from_pdf(file)
        else:
            raise ValueError("Invalid file type")

        
        result = self.model(doc)


        results = [
        OCROut(
            name=str(i),
            orientation=page.orientation,
            language=page.language,
            dimensions=page.dimensions,
            items=[
                OCRPage(
                    blocks=[
                        OCRBlock(
                            geometry=self.resolve_geometry(block.geometry),
                            objectness_score=round(block.objectness_score, 2),
                            lines=[
                                OCRLine(
                                    geometry=self.resolve_geometry(line.geometry),
                                    objectness_score=round(line.objectness_score, 2),
                                    words=[
                                        OCRWord(
                                            value=word.value,
                                            geometry=self.resolve_geometry(word.geometry),
                                            objectness_score=round(word.objectness_score, 2),
                                            confidence=round(word.confidence, 2),
                                            crop_orientation=word.crop_orientation,
                                        )
                                        for word in line.words
                                    ],
                                )
                                for line in block.lines
                            ],
                        )
                        for block in page.blocks
                    ]
                )
            ],
            )
            for i, page in enumerate(result.pages)
        ]

        return results

    def resolve_geometry(
        self,
        geom: Any,
    ) -> (
        tuple[float, float, float, float]
        | tuple[float, float, float, float, float, float, float, float]
    ):
        if len(geom) == 4:
            return (*geom[0], *geom[1], *geom[2], *geom[3])
        return (*geom[0], *geom[1])
