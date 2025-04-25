import os
import base64
from io import BytesIO

import cv2
import torch
import numpy as np
import ultralytics
from PIL import Image
from PIL import ImageEnhance

from loguru import logger   

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class SignatureDetector:
    def __init__(self) -> None:
        self.model = ultralytics.YOLO(
            os.path.join(CURRENT_DIR, "best.pt")
        )

        if torch.cuda.is_available():
            self.model.to("cuda").eval()
            logger.info("Using GPU for OCR")
        else:
            self.model.to("cpu").eval()
            logger.info("Using CPU for OCR")

    def detect_signatures(self, image: np.ndarray) -> int:
        """ "
        Method to detect the number of signatures in an image.
        """
        self.model.args["data"] = (
            os.path.join(CURRENT_DIR, "data.yaml")
        )
        result = self.model(image, conf=0.18, iou=0.3)[0]
        df = result.to_df()

        if len(df) == 0:
            return 0
        else:
            return len(df)

    def extract_signatures(self, image: np.ndarray) -> list:
        """ "
        Method to extract the signatures from an image.
        """
        self.model.args["data"] = "data.yaml"
        result = self.model(image, conf=0.18, iou=0.3)[0]
        df = result.to_df()
        signatures = []

        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        
        for _, row in df.iterrows():
            if row["class"] == 0:
                bbox_data = row["box"]
                x0, y0, x1, y1 = bbox_data.values()
                signature = img[int(y0) : int(y1), int(x0) : int(x1)]
                signature_base64 = self.img_to_base64(Image.fromarray(signature))
                signatures.append(signature_base64[0])

        return signatures


    def img_to_base64(self, image, max_size: int = 2500) -> str:
        """
        Convert an image to base64 encoding.

        Parameters:
        -----------
        - `image_path`: Path to the image file.
        - `max_size`: Maximum size of the image in any dimension. Default is 3000.

        Returns:
        --------
        - `base64_image`: Base64 encoding of the image.
        """

        img = [image]

        for i in range(len(img)):
            if max(img[i].size) > max_size:
                max_dim = max(img[i].size)
                ratio = max_size / max_dim
                img[i] = img[i].resize(
                    (int(img[i].size[0] * ratio), int(img[i].size[1] * ratio))
                )

            img[i] = img[i].convert("RGB")

            enhancer = ImageEnhance.Contrast(img[i])
            img[i] = enhancer.enhance(3.5)

            enhancer = ImageEnhance.Sharpness(img[i])
            img[i] = enhancer.enhance(3.5)

        imgs_base64 = []
        for i in range(len(img)):
            buffered = BytesIO()
            img[i].save(buffered, format="JPEG")
            imgs_base64.append(
                base64.b64encode(buffered.getvalue()).decode("utf-8")
            )

        return imgs_base64