import os
import time

import torch
import numpy as np
from torch import nn
from PIL import Image, ImageOps
from torchvision import models, transforms

from app.infrastructure.ml.aligner.utils.page_extractor import PageExtractor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    """
    Construye un modelo ResNet-50 con fine-tuning en las últimas capas.

    Returns:
        model: El modelo listo para ser entrenado.
    """
    # Cargar el modelo base preentrenado en ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Congelar las capas del modelo base
    for param in model.parameters():
        param.requires_grad = False

    # Modificar la última capa para clasificación binaria
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1),
        nn.Sigmoid(),  # Salida binaria
    )

    return model.to(device)


class AlignerModel(object):
    """
    Aligner model based on the Recursive-CNNs architecture.
    """
    def __init__(
        self, 
        model_path: str = os.path.join(CURRENT_DIR, "checkpoints/resnet50_finetuned_2.pth"),
        corner_doc_path: str = os.path.join(CURRENT_DIR, "checkpoints/Experiment-12-docdocument_resnet.pb"),
        corner_refiner_path: str = os.path.join(CURRENT_DIR, "checkpoints/corner-refinement-experiment-4corner_resnet.pb")
    ) -> None:
        """
        Args
        ----
        model_path : str
            Path to the trained model that predicts if the image needs to be aligned or not.
        corner_doc_path : str
            Path to the trained model that predicts the corners of the document.
        corner_refiner_path : str
            Path to the trained model that refines the corners of the document.
        """
        self.corner_doc_path = corner_doc_path
        self.corner_refiner_path = corner_refiner_path
        self.cuda = torch.cuda.is_available()
        self.extractor = PageExtractor(
            self.corner_refiner_path, self.corner_doc_path
        )

        self.model = build_model()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def align_image(self, image_array: np.ndarray) -> Image.Image:
        """
        Aligns the image if it is necessary.

        Args
        ----
        image_array : np.ndarray
            Numpy array representation of the image to be aligned.
        
        Returns
        ----
        Image.Image
            Aligned image if alignment was necessary, otherwise the original image.
        """
        t0_total = time.time()
        # Keep original image for return if no alignment needed
        original_img = Image.fromarray(image_array).convert("RGB")
        
        # Check if GPU is available
        use_gpu = torch.cuda.is_available()
        
        # Process image for model prediction - avoid unnecessary copies
        img = ImageOps.exif_transpose(original_img)
        img = self.transform(img)
        img = img.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            if use_gpu:
                # Move to GPU and use mixed precision for faster processing
                img = img.cuda(non_blocking=True)
                with torch.cuda.amp.autocast():
                    t0 = time.time()
                    output = self.model(img)
                # Get result from GPU
                output = output.cpu().numpy()[0]
            else:
                t0 = time.time()
                output = self.model(img).numpy()[0]
                
            prediction = 1 if output > 0.5 else 0
            
        # Process according to prediction
        if prediction == 1:
            # Align the image
            t0 = time.time()
            _, warped = self.extractor.extract_document(image_array, 0.90)
            
            # Clear GPU memory
            if use_gpu:
                torch.cuda.empty_cache()
                
            return warped
        else:
            # Clear GPU memory
            if use_gpu:
                torch.cuda.empty_cache()
                
            return original_img
