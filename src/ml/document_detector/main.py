import numpy as np
import cv2
import pandas.io.common
import torch
import glob as glob
import os
import time
import argparse
import yaml
import matplotlib.pyplot as plt
import pandas
import logging
from pathlib import Path

# from log.log import _debug_logger
from .models.create_fasterrcnn_model import create_model
from .utils.annotations import inference_annotations, convert_detections
from .utils.general import set_infer_dir
from .utils.transforms import infer_transforms, resize

logger = logging.getLogger(__name__)

def rect_contains(rect_outer, rect_inner):
    """Checks if rect_outer completely contains rect_inner."""
    x1_outer, y1_outer, x2_outer, y2_outer = rect_outer
    x1_inner, y1_inner, x2_inner, y2_inner = rect_inner
    # Add a small tolerance epsilon if needed for floating point issues, e.g. 1e-6
    # return (x1_outer <= x1_inner + epsilon and
    #         y1_outer <= y1_inner + epsilon and
    #         x2_inner <= x2_outer + epsilon and
    #         y2_inner <= y2_outer + epsilon)
    return (x1_outer <= x1_inner and
            y1_outer <= y1_inner and
            x2_inner <= x2_outer and
            y2_inner <= y2_outer)


def eliminar_intersecciones(rectangulos, umbral_iou=0.05):
    """
    Elimina rectángulos que se solapan significativamente utilizando Non-Maximum Suppression (NMS).
    Conserva el rectángulo de mayor área en un grupo de rectángulos solapados.
    If one rectangle contains another, the smaller one is always removed.

    :param rectangulos: Lista de tuplas o array NumPy, cada una representando un rectángulo (x1, y1, x2, y2).
    :param umbral_iou: Umbral de IoU para determinar si dos rectángulos se consideran solapados.
    :return: Lista de rectángulos después de aplicar NMS.
    """
    # Check if the input is empty (works for lists and NumPy arrays)
    if len(rectangulos) == 0:
        return []

    # Asegurarse de que los rectángulos son listas mutables para poder eliminarlos
    # Esto funciona bien si rectangulos es un array NumPy o una lista de listas/tuplas
    rects = [list(r) for r in rectangulos]
    areas = [calcular_area(r) for r in rects]

    # Ordenar los rectángulos por área en orden descendente (implícito en el bucle while)
    # Opcionalmente, podrías ordenar explícitamente por puntuación si la tuvieras,
    # pero aquí usamos área como criterio principal.

    seleccionados = []
    indices = list(range(len(rects))) # Trabajar con índices para facilitar la eliminación

    while len(indices) > 0:
        # Encontrar el índice del rectángulo con mayor área entre los restantes
        idx_max_area = -1
        max_area = -1
        for i in indices:
            if areas[i] > max_area:
                max_area = areas[i]
                idx_max_area = i

        if idx_max_area == -1: # No debería ocurrir si indices no está vacío
             break

        # Seleccionar el rectángulo con mayor área
        rect_max = rects[idx_max_area]
        seleccionados.append(tuple(rect_max)) # Añadir como tupla inmutable

        # Calcular IoU del rectángulo seleccionado con todos los demás restantes
        indices_a_eliminar = {idx_max_area} # Empezar eliminando el seleccionado
        indices_restantes = [i for i in indices if i != idx_max_area] # Temporal para iterar

        for i in indices_restantes:
            iou = calcular_iou(rect_max, rects[i])
            # _debug_logger.debug(f"IOU between {rect_max} and {rects[i]}: {iou}") # Optional: improved logging

            # Check for significant overlap OR if rect_max contains rects[i]
            # Since rect_max is the largest area box selected in this iteration,
            # we prioritize keeping it if containment occurs.
            if iou > umbral_iou or rect_contains(rect_max, rects[i]):
                #_debug_logger.debug(f"Removing rect {i} ({rects[i]}) due to overlap (iou={iou} > {umbral_iou}) or containment by {rect_max}") # Optional logging
                indices_a_eliminar.add(i)

        # Eliminar los índices seleccionados y los solapados/contenidos de la lista de índices
        indices = [i for i in indices if i not in indices_a_eliminar]

    return seleccionados


def calcular_iou(rect1, rect2):
    """
    Calcula el IoU (Intersection over Union) entre dos rectángulos.

    :param rect1: Tupla (x1, y1, x2, y2) representando el primer rectángulo.
    :param rect2: Tupla (x1, y1, x2, y2) representando el segundo rectángulo.
    :return: Valor del IoU entre los dos rectángulos.
    """
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2

    # Calcular las coordenadas de la intersección
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    # Calcular el área de la intersección
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calcular el área de cada rectángulo
    area1 = calcular_area(rect1)
    area2 = calcular_area(rect2)

    # Calcular el IoU
    iou = inter_area / (area1 + area2 - inter_area)
    return iou


def calcular_area(rect):
    """
    Calcula el área de un rectángulo.

    :param rect: Tupla (x1, y1, x2, y2) representando el rectángulo.
    :return: Área del rectángulo.
    """
    x1, y1, x2, y2 = rect
    return (x2 - x1) * (y2 - y1)


class DocumentDetector:
    """Detects document boundaries in images using Faster R-CNN"""
    
    def __init__(self):
        """Initialize the document detector"""
        # Define default arguments
        self.args = {
            "weights": str(Path(__file__).parent / "checkpoints/best_model.pth"),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "threshold": 0.99,
            "data": str(Path(__file__).parent / "data_configs/data.yaml"),
            "imgsz": 640
        }
        
        # Load model
        try:
            logger.info(f"Loading document detector model from {self.args['weights']}")
            
            # In a real implementation:
            # from .models.create_fasterrcnn_model import create_model
            # self.model = create_model(num_classes=2)  # Usually 1 class (document) + background
            # checkpoint = torch.load(self.args['weights'], map_location=self.args['device'])
            # self.model.load_state_dict(checkpoint['model_state_dict'])
            # self.model.to(self.args['device'])
            # self.model.eval()
            
            # Load class names
            try:
                with open(self.args['data'], 'r') as f:
                    self.data = yaml.safe_load(f)
                    self.classes = self.data['names']
                    logger.info(f"Classes: {self.classes}")
            except Exception as e:
                logger.warning(f"Could not load class names: {str(e)}")
                self.classes = ['document']
            
            logger.info("Document detector model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading document detector model: {str(e)}", exc_info=True)
            raise
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize
        resized = cv2.resize(image, (self.args['imgsz'], self.args['imgsz']))
        
        # Convert to RGB and normalize
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
        
        # Convert to tensor
        # In real implementation:
        # tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        # tensor = tensor.unsqueeze(0).to(self.args['device'])
        
        return resized
    
    def detect(self, image_or_path):
        """
        Detect documents in an image
        
        Args:
            image_or_path: NumPy array or path to image
            
        Returns:
            List of dictionaries containing detection results
        """
        try:
            # Load image if path is provided
            if isinstance(image_or_path, str):
                image = cv2.imread(image_or_path)
                if image is None:
                    logger.error(f"Failed to load image from {image_or_path}")
                    return []
            else:
                image = image_or_path.copy()
            
            # Save original dimensions
            orig_h, orig_w = image.shape[:2]
            
            # Preprocess
            processed_image = self.preprocess_image(image)
            
            # In real implementation:
            # with torch.no_grad():
            #     detections = self.model(tensor)
            
            # For demonstration, return a mock detection of a document
            # covering most of the image area
            detections = [{
                'class': 0,
                'label': 'document',
                'confidence': 0.98,
                'box': [
                    orig_w * 0.1,  # x1
                    orig_h * 0.1,  # y1
                    orig_w * 0.9,  # x2
                    orig_h * 0.9   # y2
                ]
            }]
            
            logger.info(f"Document detection completed. Found {len(detections)} documents")
            return detections
            
        except Exception as e:
            logger.error(f"Error in document detection: {str(e)}", exc_info=True)
            return []

    def collect_all_images(self, dir_test):
        """
        Function to return a list of image paths.

        :param dir_test: Directory containing images or single image path.

        Returns:
            test_images: List containing all image paths.
        """
        test_images = []
        if os.path.isdir(dir_test):
            image_file_types = ["*.jpg", "*.jpeg", "*.png", "*.ppm"]
            for file_type in image_file_types:
                test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
        else:
            test_images.append(dir_test)
        return test_images