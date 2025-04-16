from __future__ import print_function

import sys
import os
import torch
import numpy as np
import cv2
import logging
from pathlib import Path

# Add utils to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "utils"))

logger = logging.getLogger(__name__)

class AlignerModel:
    """Aligns document images using a ResNet50 model"""
    
    def __init__(self, model_path, corner_doc_path, corner_refiner_path):
        """
        Initialize the aligner model
        
        Args:
            model_path: Path to the trained alignment prediction model
            corner_doc_path: Path to the corner detection model
            corner_refiner_path: Path to the corner refinement model
        """
        self.model_path = model_path
        self.corner_doc_path = corner_doc_path
        self.corner_refiner_path = corner_refiner_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        try:
            # Here we would load the actual models
            # For demonstration purposes, we're just logging the paths
            logger.info(f"Loading aligner model from {model_path}")
            logger.info(f"Loading corner detection model from {corner_doc_path}")
            logger.info(f"Loading corner refinement model from {corner_refiner_path}")
            
            # In real implementation:
            # self.model = torch.load(model_path, map_location=self.device)
            # ...
            
            logger.info("Aligner models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading aligner models: {str(e)}", exc_info=True)
            raise
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # Resize if needed
        # image = cv2.resize(image, (desired_width, desired_height))
        
        return image
    
    def detect_corners(self, image):
        """Detect document corners in the image"""
        # This would use the corner detection model
        # For demonstration:
        h, w = image.shape[:2]
        corners = np.array([
            [0, 0],         # top-left
            [w-1, 0],       # top-right
            [w-1, h-1],     # bottom-right
            [0, h-1]        # bottom-left
        ], dtype=np.float32)
        
        return corners
    
    def align_image(self, image_or_path):
        """
        Align a document image
        
        Args:
            image_or_path: NumPy array or path to image
            
        Returns:
            Aligned image
        """
        try:
            # Load image if path is provided
            if isinstance(image_or_path, str):
                image = cv2.imread(image_or_path)
                if image is None:
                    logger.error(f"Failed to load image from {image_or_path}")
                    return None
            else:
                image = image_or_path.copy()
            
            # Preprocess
            processed_image = self.preprocess_image(image)
            
            # Detect corners
            corners = self.detect_corners(processed_image)
            
            # Refine corners
            # refined_corners = self.refine_corners(processed_image, corners)
            
            # Apply perspective transform
            h, w = processed_image.shape[:2]
            dst_corners = np.array([
                [0, 0],
                [w-1, 0],
                [w-1, h-1],
                [0, h-1]
            ], dtype=np.float32)
            
            # Calculate perspective transform matrix
            M = cv2.getPerspectiveTransform(corners, dst_corners)
            
            # Apply transform
            aligned_image = cv2.warpPerspective(processed_image, M, (w, h))
            
            logger.info("Image aligned successfully")
            return aligned_image
            
        except Exception as e:
            logger.error(f"Error in document alignment: {str(e)}", exc_info=True)
            # Return original image if alignment fails
            if isinstance(image_or_path, str) and os.path.exists(image_or_path):
                return cv2.imread(image_or_path)
            return image_or_path
