import os
import time
import yaml
import glob as glob

import cv2
import torch
import numpy as np
from fastapi import UploadFile

from .utils.annotations import convert_detections
from .utils.transforms import infer_transforms, resize
from .models.create_fasterrcnn_model import create_model

from loguru import logger

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class DocumentDetector:
    def __init__(self):
        self.args = {
            "model": None,
            "weights": os.path.join(CURRENT_DIR, "checkpoints/best_model.pth"),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "threshold": 0.99,
            "data": os.path.join(CURRENT_DIR, "data_configs/data.yaml"),
            "output": None,
            "imgsz": None,
            "square_img": False,
            "classes": None,
        }
        self.model = None
        self.NUM_CLASSES = None
        self.CLASSES = None
        self.COLORS = None
        self.DEVICE = self.args["device"]
        
        self.data_configs = None
        if self.args["data"] is not None:
            with open(self.args["data"]) as file:
                self.data_configs = yaml.safe_load(file)
            self.NUM_CLASSES = self.data_configs["NC"]
            self.CLASSES = self.data_configs["CLASSES"]
        
        self._load_model()
        
        np.random.seed(42)
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

    def _load_model(self):
        """Carga el modelo de detección de objetos"""
        if self.args["weights"] is None:
            if self.data_configs is None:
                with open(
                    os.path.join("data_configs", "test_image_config.yaml")
                ) as file:
                    self.data_configs = yaml.safe_load(file)
                self.NUM_CLASSES = self.data_configs["NC"]
                self.CLASSES = self.data_configs["CLASSES"]
            try:
                build_model = create_model[self.args["model"]]
                self.model, _ = build_model(
                    num_classes=self.NUM_CLASSES, coco_model=True
                )
            except:
                build_model = create_model["fasterrcnn_resnet50_fpn_v2"]
                self.model, _ = build_model(
                    num_classes=self.NUM_CLASSES, coco_model=True
                )
        # Cargar pesos si se proporciona una ruta.
        if self.args["weights"] is not None:
            checkpoint = torch.load(self.args["weights"], map_location=self.DEVICE)
            if self.data_configs is None:
                self.data_configs = True
                self.NUM_CLASSES = checkpoint["data"]["NC"]
                self.CLASSES = checkpoint["data"]["CLASSES"]
            try:
                build_model = create_model[str(self.args["model"])]
            except:
                build_model = create_model[checkpoint["model_name"]]
            self.model = build_model(num_classes=self.NUM_CLASSES, coco_model=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        self.model.to(self.DEVICE).eval()
        if torch.cuda.is_available():
            self.model.half()
            logger.info("Using GPU for OCR")
        else:
            logger.info("Using CPU for OCR")

    def detect(self, input):
        test_images = [input]
        detection_threshold = self.args["threshold"]
        frame_count = 0
        total_fps = 0
        results = []

        for i in range(len(test_images)):
            if isinstance(test_images[i], UploadFile) or hasattr(test_images[i], 'read'):
                file_bytes = test_images[i].file.read() if hasattr(test_images[i], 'file') else test_images[i].read()
                nparr = np.frombuffer(file_bytes, np.uint8)
                orig_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # Read from path
                orig_image = cv2.imread(test_images[i])
            _, frame_width, _ = orig_image.shape
            if self.args["imgsz"] != None:
                RESIZE_TO = self.args["imgsz"]
            else:
                RESIZE_TO = frame_width
            
            image_resized = resize(
                orig_image, RESIZE_TO, square=self.args["square_img"]
            )
            image = image_resized.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = infer_transforms(image)
            image = torch.unsqueeze(image, 0)
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(image.to(self.DEVICE))
            end_time = time.time()

            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1
            outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

            # Carry further only if there are detected boxes.
            if len(outputs[0]["boxes"]) != 0:
                draw_boxes, *_ = convert_detections(
                    outputs, detection_threshold, self.CLASSES, self.args
                )
                if len(draw_boxes) == 0:
                    return []

                height, width, _ = orig_image.shape
            
                draw_boxes = self.eliminar_intersecciones(draw_boxes)
                                
                if len(draw_boxes) == 1:
                    return [orig_image]
                
                for box in draw_boxes:
                    xmin, ymin, xmax, ymax = box
                    xmin, ymin, xmax, ymax = (
                        int(xmin / image_resized.shape[1] * width),
                        int(ymin / image_resized.shape[0] * height),
                        int(xmax / image_resized.shape[1] * width),
                        int(ymax / image_resized.shape[0] * height),
                    )

                    cut_image = orig_image[ymin:ymax, xmin:xmax]
                    results.append(cut_image)
                
                return results
            else:
                return []
            
    def rect_contains(self, rect_outer, rect_inner):
        """Checks if rect_outer completely contains rect_inner."""
        x1_outer, y1_outer, x2_outer, y2_outer = rect_outer
        x1_inner, y1_inner, x2_inner, y2_inner = rect_inner
        return (x1_outer <= x1_inner and
                y1_outer <= y1_inner and
                x2_inner <= x2_outer and
                y2_inner <= y2_outer)


    def eliminar_intersecciones(self, rectangulos, umbral_iou=0.05):
        """
        Elimina rectángulos que se solapan significativamente utilizando Non-Maximum Suppression (NMS).
        Conserva el rectángulo de mayor área en un grupo de rectángulos solapados.
        If one rectangle contains another, the smaller one is always removed.

        :param rectangulos: Lista de tuplas o array NumPy, cada una representando un rectángulo (x1, y1, x2, y2).
        :param umbral_iou: Umbral de IoU para determinar si dos rectángulos se consideran solapados.
        :return: Lista de rectángulos después de aplicar NMS.
        """
        if len(rectangulos) == 0:
            return []

        rects = [list(r) for r in rectangulos]
        areas = [self.calcular_area(r) for r in rects]

        seleccionados = []
        indices = list(range(len(rects)))

        while len(indices) > 0:
            idx_max_area = -1
            max_area = -1
            for i in indices:
                if areas[i] > max_area:
                    max_area = areas[i]
                    idx_max_area = i

            if idx_max_area == -1:
                break

            rect_max = rects[idx_max_area]
            seleccionados.append(tuple(rect_max))

            indices_a_eliminar = {idx_max_area}
            indices_restantes = [i for i in indices if i != idx_max_area]

            for i in indices_restantes:
                iou = self.calcular_iou(rect_max, rects[i])
                if iou > umbral_iou or self.rect_contains(rect_max, rects[i]):
                    indices_a_eliminar.add(i)

            indices = [i for i in indices if i not in indices_a_eliminar]

        return seleccionados


    def calcular_iou(self, rect1, rect2):
        """
        Calcula el IoU (Intersection over Union) entre dos rectángulos.

        :param rect1: Tupla (x1, y1, x2, y2) representando el primer rectángulo.
        :param rect2: Tupla (x1, y1, x2, y2) representando el segundo rectángulo.
        :return: Valor del IoU entre los dos rectángulos.
        """
        x1_1, y1_1, x2_1, y2_1 = rect1
        x1_2, y1_2, x2_2, y2_2 = rect2

        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        area1 = self.calcular_area(rect1)
        area2 = self.calcular_area(rect2)

        iou = inter_area / (area1 + area2 - inter_area)
        return iou


    def calcular_area(self, rect):
        """
        Calcula el área de un rectángulo.

        :param rect: Tupla (x1, y1, x2, y2) representando el rectángulo.
        :return: Área del rectángulo.
        """
        x1, y1, x2, y2 = rect
        return (x2 - x1) * (y2 - y1)