FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3-pip git curl \
    build-essential libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    libgeos-dev supervisor \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as default
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Install uv (a faster Python package installer)
RUN curl -sSL https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    chmod +x /usr/local/bin/uv && \
    uv --version

# Install setuptools separately first to avoid version conflicts
RUN /usr/local/bin/uv pip install --system setuptools wheel --no-cache

# Install gunicorn for serving the application
RUN /usr/local/bin/uv pip install --system gunicorn --no-cache

# Install paddlepaddle with retry logic (using uv instead of pip)
RUN for i in $(seq 1 3); do \
    /usr/local/bin/uv pip install --system paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html --no-cache && break || \
    if [ $i -eq 3 ]; then exit 1; fi; \
    echo "Retrying paddlepaddle installation..."; \
    sleep 5; \
    done

# Install shapely first
RUN /usr/local/bin/uv pip install --system shapely --no-cache

# Copy and install requirements
COPY ./requirements.txt /app
RUN /usr/local/bin/uv pip install --system -r requirements.txt --no-cache --index-strategy unsafe-best-match

# Initialize PaddleOCR
RUN python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=True, lang='en', det_lang='ml', show_log=False, use_gpu=True)"

# Install pyzbar
RUN /usr/local/bin/uv pip install --system pyzbar --no-cache
RUN apt-get update && apt-get install -y --no-install-recommends libzbar0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN /usr/local/bin/uv pip install --system "python-doctr[torch]" --no-cache

# Set environment variable for protobuf compatibility
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Downgrade protobuf package to avoid compatibility issues
RUN /usr/local/bin/uv pip install --system protobuf==3.20.3 --no-cache

# Install deps
RUN apt-get update && apt-get install -y --no-install-recommends libpangocairo-1.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install nltk
RUN /usr/local/bin/uv pip install --system nltk --no-cache
RUN python -c "import nltk; nltk.download('stopwords')"

# Install pdf2image
RUN /usr/local/bin/uv pip install --system pdf2image --no-cache


# Copy application code
COPY . /app

# Create logs directory
RUN mkdir -p /app/logs && chmod 777 /app/logs

# Create supervisor configuration
RUN mkdir -p /etc/supervisor/conf.d
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8000

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"] 