from typing import Literal, Any, List, Dict, Tuple, Union

import torch
import gc
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from app.schemas.schemas import OCROut, OCRPage, OCRBlock, OCRLine, OCRWord

import time
from loguru import logger


class DocTRRunner:
    _instance = None
    _model = None
    
    def __new__(cls):
        # Implement a singleton pattern to ensure we only have one model in memory
        if cls._instance is None:
            cls._instance = super(DocTRRunner, cls).__new__(cls)
        return cls._instance
        
    def __init__(self):
        # Only initialize the model once
        if DocTRRunner._model is None:
            logger.info("Initializing DocTR model...")
            DocTRRunner._model = self._initialize_model()
        self.model = DocTRRunner._model
        
    def _initialize_model(self):
        # Initialize the model with optimized settings
        model = ocr_predictor(pretrained=True)
        
        if torch.cuda.is_available():
            # Use mixed precision for better performance
            model.to("cuda").eval().half()
            
            # # Compile using TensorRT if available
            # model = torch.compile(
            #     model,
            #     backend='torch_tensorrt',
            #     options={
            #         "enabled_precisions": {torch.float16},
            #     }
            # )
            
            # Set lower precision for inference to save memory
            with torch.cuda.amp.autocast():
                # Warm up the model with a dummy input to precompile CUDA kernels
                dummy_input = torch.zeros((1, 3, 224, 224), device="cuda", dtype=torch.float16)
                # Dummy forward pass to initialize CUDA kernels
                _ = model.det_predictor.model(dummy_input)
                
            logger.info("DocTR model initialized on GPU with FP16 precision")
        else:
            model.to("cpu").eval()
            logger.info("DocTR model initialized on CPU")
            
        return model

    def render(
        self, file: bytes, file_type: Literal["image", "pdf"]
    ) -> List[Dict]:
        start_time = time.time()
        if file_type in ["jpg", "jpeg", "png"]:
            doc = DocumentFile.from_images([file])
        elif file_type == "pdf":
            doc = DocumentFile.from_pdf(file)
        else:
            raise ValueError("Invalid file type")

        try:
            start_time = time.time()
            
            # Use mixed precision for inference
            with torch.cuda.amp.autocast():
                result = self.model(doc).render()
                
            # Clear GPU cache after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            end_time = time.time()
            logger.info(f"Time taken to render document: {end_time - start_time} seconds")
            return result
        except Exception as e:
            raise ValueError(f"Error rendering document: {e}")

    def ocr(
        self, file: bytes, file_type: Literal["image", "pdf"]
    ) -> List[Dict]:
        if file_type in ["jpg", "jpeg", "png"]:
            doc = DocumentFile.from_images([file])
        elif file_type == "pdf":
            doc = DocumentFile.from_pdf(file)
        else:
            raise ValueError("Invalid file type")

        # Use mixed precision for inference
        with torch.cuda.amp.autocast():
            result = self.model(doc)
            
        # Clear GPU cache after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        Union[Tuple[float, float, float, float],  Tuple[float, float, float, float, float, float, float, float]]
    ):
        if len(geom) == 4:
            return (*geom[0], *geom[1], *geom[2], *geom[3])
        return (*geom[0], *geom[1])
