# Base image with CUDA support
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libzbar0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Link python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements file
COPY requirements.txt .

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --system --no-cache-dir && \
    uv pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html --no-cache-dir --system && \
    uv pip install -r requirements.txt --system --no-cache-dir

# Install nltk data
RUN python -c "import nltk; nltk.download('stopwords')"

# Install Paddle model
RUN python -c "from paddleocr import PaddleOCR; _ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, det_lang='ml', show_log=False)"

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "10"]