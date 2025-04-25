from typing import List

import paddle
import numpy as np
from paddleocr import PaddleOCR
from app.domain.entities import PageEntity, ItemEntity


class PaddleOCRRunner:
    def __init__(self):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            det_lang="ml",
            show_log=False,
            use_gpu=paddle.device.is_compiled_with_cuda(),
        )

    def predict(self, image: np.ndarray) -> List[dict]:
        return self.ocr.ocr(image, cls=True)

    @staticmethod
    def process_result(
        ocr_results: List[dict], width: int, height: int
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
