# Base image with CUDA support
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
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


# If not, install python3.10
RUN if ! command -v python3.10 &> /dev/null; then \
    apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.10 && \
    rm -rf /var/lib/apt/lists/*; \
    fi

# Verify python3.10 is installed
RUN python3.10 --version

# Link python3.10 to python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Copy requirements file
COPY requirements.txt .

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Install dependencies
RUN uv pip install --no-cache-dir --upgrade pip --system && \
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --system --no-cache-dir && \
    uv pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html --no-cache-dir --system && \
    uv pip install -r requirements.txt --system --no-cache-dir

# Install nltk data
RUN python -c "import nltk; nltk.download('stopwords')"

# Install Paddle model
RUN python -c "from paddleocr import PaddleOCR; _ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, det_lang='ml', show_log=False)"

# Install libpangocairo
RUN apt-get update && apt-get install -y libpangocairo-1.0-0

# Install doctr model
RUN python -c "from doctr.models import ocr_predictor; _ocr = ocr_predictor(pretrained=True)"

RUN pip install orjson

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "10"]