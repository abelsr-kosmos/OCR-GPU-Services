from typing import List

import paddle
import numpy as np
from paddleocr import PaddleOCR
from app.domain.entities import PageEntity, ItemEntity

from loguru import logger


class PaddleOCRRunner:
    def __init__(self):
        # Check if CUDA is available
        use_gpu = paddle.device.is_compiled_with_cuda()
        gpu_mem = 2000  # Default GPU memory limit in MB
        
        if use_gpu:
            # Set better GPU memory management
            try:
                # Get available GPU memory
                gpu_info = paddle.device.cuda.get_device_properties(0)
                total_memory = gpu_info.total_memory / (1024 * 1024)  # Convert to MB
                # Use 80% of available memory at most
                gpu_mem = int(total_memory * 0.8)
            except:
                # Fallback to default if we can't get GPU info
                pass
                
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            det_lang="ml",
            show_log=False,
            use_gpu=use_gpu,
            gpu_mem=gpu_mem,
            enable_mkldnn=not use_gpu,  # Enable MKL-DNN optimization for CPU
            cpu_threads=8 if not use_gpu else 1,  # More CPU threads if on CPU
            # use_tensorrt=True,
            precision="fp16"
            # trt_calib_mode=True,
            # use_static=False,
        )

    def predict(self, image: np.ndarray) -> List[dict]:
        try:
            return self.ocr.ocr(image, cls=True)
        except Exception as e:
            logger.error(f"Error during OCR prediction: {e}")
            raise RuntimeError(f"Error during OCR prediction: {e}")

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
