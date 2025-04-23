import paddle
import numpy as np
import os
from pathlib import Path
from loguru import logger
from paddleocr import PaddleOCR

from app.domain.entities import PageEntity, ItemEntity

# Get cache directory from environment or use default
PADDLE_CACHE_DIR = os.environ.get("PADDLE_CACHE_DIR", str(Path("./model_cache/paddle").absolute()))
os.makedirs(PADDLE_CACHE_DIR, exist_ok=True)
logger.info(f"Using PaddleOCR cache directory: {PADDLE_CACHE_DIR}")

class PaddleOCRRunner:
    def __init__(self):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            det_lang="ml",
            show_log=False,
            use_gpu=paddle.device.is_compiled_with_cuda(),
            det_model_dir=os.path.join(PADDLE_CACHE_DIR, "det_models"),
            rec_model_dir=os.path.join(PADDLE_CACHE_DIR, "rec_models"),
            cls_model_dir=os.path.join(PADDLE_CACHE_DIR, "cls_models")
        )

    def predict(self, image: np.ndarray) -> list[dict]:
        return self.ocr.ocr(image, cls=True)

    @staticmethod
    def process_result(
        ocr_results: list[dict], width: int, height: int
    ) -> PageEntity:
        pages = []
        texts = ""

        for result in ocr_results:
            json_output = []
            for page in result:
                for entry in page:
                    coordinates = entry[0]
                    coordinates = [
                        list(map(int, coord)) for coord in coordinates
                    ]
                    text, confidence = entry[1]
                    json_output.append(
                        {
                            "Coordinates": coordinates,
                            "Text": text,
                            "Score": confidence,
                        }
                    )
                    texts += f"{text} "
            pages.append(json_output)

        result = dict(
            pages=pages, text=texts.split(" "), width=width, height=height
        )

        result = PageEntity(
            items=[
                ItemEntity(
                    coordinates=item["Coordinates"],
                    text=item["Text"],
                    score=item["Score"],
                )
                for item in result["pages"]
            ],
            text=result["text"],
            width=result["width"],
            height=result["height"],
        )

        return result
