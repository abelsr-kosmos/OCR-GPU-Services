#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Install ML Models
python -c "import nltk; nltk.download('stopwords'); from doctr.models import ocr_predictor; _ocr = ocr_predictor(pretrained=True); from paddleocr import PaddleOCR; _ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, det_lang='ml', show_log=False); from qreader import QReader; _qr = QReader(); from app.infrastructure.ml.aligner.main import AlignerModel; _aligner = AlignerModel(); from app.infrastructure.ml.signature_service import SignatureService; _signature_service = SignatureService()"

# Download necessary weights and models
wget https://download.pytorch.org/models/resnet50-0676ba61.pth -P /root/.cache/torch/hub/checkpoints/
wget https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth -P /root/.cache/torch/hub/checkpoints/

# Start the server
gunicorn app.main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 4