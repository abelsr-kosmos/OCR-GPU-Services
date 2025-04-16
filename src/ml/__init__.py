import logging
import os
from pathlib import Path
from .aligner.main import AlignerModel
from .classifier.model import OCRClassifier
from .document_detector.main import DocumentDetector
from .signature_detector.main import SignatureDetector

logger = logging.getLogger(__name__)

# Base path for model checkpoints
_BASE_PATH = Path(__file__).parent

# Default model paths
_ALIGNER_MODEL_PATH = _BASE_PATH / "aligner/checkpoints/resnet50_finetuned_2.pth"
_CORNER_DOC_PATH = _BASE_PATH / "aligner/checkpoints/Experiment-12-docdocument_resnet.pb"
_CORNER_REFINER_PATH = _BASE_PATH / "aligner/checkpoints/corner-refinement-experiment-4corner_resnet.pb"

# Initialize all models
try:
    logger.info("Initializing ML models")
    
    # Initialize classifier
    _classifier = OCRClassifier(
        zscore_threshold=2,
        cost_threshold=2,
        max_cost=1000,
        apply_log2cost=True,
        n_jobs=-1
    )
    
    # Initialize aligner if model files exist
    if os.path.exists(_ALIGNER_MODEL_PATH) and os.path.exists(_CORNER_DOC_PATH) and os.path.exists(_CORNER_REFINER_PATH):
        _aligner = AlignerModel(
            str(_ALIGNER_MODEL_PATH),
            str(_CORNER_DOC_PATH),
            str(_CORNER_REFINER_PATH)
        )
    else:
        logger.warning("Aligner model files not found, using placeholder")
        _aligner = AlignerModel(
            "placeholder_path1",
            "placeholder_path2",
            "placeholder_path3"
        )
    
    # Initialize document detector
    _document_detector = DocumentDetector()
    
    # Initialize signature detector
    _signature_detector = SignatureDetector()
    
    logger.info("ML models initialized successfully")
except Exception as e:
    logger.error(f"Error initializing ML models: {str(e)}", exc_info=True)
    # Create placeholder models to avoid breaking the application
    _classifier = OCRClassifier()
    _aligner = AlignerModel("placeholder_path1", "placeholder_path2", "placeholder_path3")
    _document_detector = DocumentDetector()
    _signature_detector = SignatureDetector()

# Create a simplified API for the models
async def classify_document(sample, reference_samples):
    """Classify document against reference samples"""
    try:
        return _classifier.predict(sample, reference_samples)
    except Exception as e:
        logger.error(f"Error in document classification: {str(e)}", exc_info=True)
        return -1, 0.0

async def detect_document(image):
    """Detect documents in an image"""
    try:
        return _document_detector.detect(image)
    except Exception as e:
        logger.error(f"Error in document detection: {str(e)}", exc_info=True)
        return []

async def detect_signature(image):
    """Detect signatures in an image"""
    try:
        return _signature_detector.detect(image)
    except Exception as e:
        logger.error(f"Error in signature detection: {str(e)}", exc_info=True)
        return []

async def align_document(image):
    """Align document image if needed"""
    try:
        return _aligner.align_image(image)
    except Exception as e:
        logger.error(f"Error in document alignment: {str(e)}", exc_info=True)
        # If alignment fails, return original image
        return image
