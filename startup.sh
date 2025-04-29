# Check if TensorRT is available
dpkg -l | grep tensorrt && echo "TensorRT is available" || echo "TensorRT is not available"

python -c "from paddleocr import PaddleOCR; _ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, det_lang='ml', show_log=False)"

gunicorn app.main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 4