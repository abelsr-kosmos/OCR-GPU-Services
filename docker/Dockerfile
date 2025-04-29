# Base image with CUDA support
FROM nvcr.io/nvidia/paddlepaddle:24.12-py3

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
    # libgl1-mesa-glx \
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

# Set python3.10 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Copy requirements file
COPY requirements.txt .

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Install dependencies
RUN uv pip install --no-cache-dir --upgrade pip --system && \
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --system --no-cache-dir && \
    uv pip install paddlepaddle-gpu==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ --no-cache-dir --system && \
    uv pip install -r requirements.txt --system --no-cache-dir

# Install nltk data
RUN python -c "import nltk; nltk.download('stopwords')"

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install libpangocairo
RUN apt-get update && apt-get install -y libpangocairo-1.0-0

# Install doctr model
RUN python -c "from doctr.models import ocr_predictor; _ocr = ocr_predictor(pretrained=True)"

# Expose port
EXPOSE 8000

# Copy startup script
COPY startup.sh .
RUN chmod +x startup.sh

# Copy project files
COPY . .

# Run the application
CMD ["./startup.sh"]