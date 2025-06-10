# syntax=docker/dockerfile:1.4

########################################
# ===== STAGE 1: BUILDER 🚧 =====
########################################
FROM nvcr.io/nvidia/paddlepaddle:24.12-py3 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 1. Instala dependencias de sistema y Python 3.12
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      software-properties-common wget git \
      python3.12 python3.12-venv python3.12-dev python3-pip \
      build-essential libzbar0 libglib2.0-0 poppler-utils \
      libpangocairo-1.0-0 ffmpeg libsm6 libxext6 libgl1 && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y python3.12 && \
    rm -rf /var/lib/apt/lists/*

# 2. Configura Python 3.12 como default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

WORKDIR /app

# 3. Copia requirements y instala UV
COPY requirements.txt ./
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"
RUN uv venv --python 3.12
RUN source .venv/bin/activate

# 4. Instala dependencias Python con cache de pip para BuildKit 🎯
RUN --mount=type=cache,target=/app/.venv/cache/pip \
    uv pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu118 \
        --no-cache-dir && \
    uv pip install paddlepaddle-gpu==3.0.0rc1 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ \
        --no-cache-dir && \
    uv pip install -r requirements.txt --no-cache-dir

# 5. Copia el código de la aplicación
COPY startup.sh ./
COPY app/ ./app/


########################################
# ===== STAGE 2: RUNTIME 🚀 =====
########################################
FROM nvcr.io/nvidia/paddlepaddle:24.12-py3

ENV PYTHONUNBUFFERED=1 \
    PYTHONBYTECODE=1

# 1. Instala sólo librerías necesarias en runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libzbar0 libglib2.0-0 libsm6 libxext6 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# 2. Copia entorno virtual y binarios de UV desde el builder
COPY --from=builder /root/.local /root/.local
ENV PATH="/root/.local/bin/:$PATH"
COPY --from=builder /app/.venv /app/.venv

WORKDIR /app
# 3. Copia startup y código (si no copiado antes)
COPY --from=builder /app/startup.sh ./
COPY --from=builder /app/app ./app

RUN chmod +x startup.sh

# 4. Instala pdf2image
RUN uv pip install pdf2image

# 5. Instalar Poppler
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# 6. Expone puerto y arranca
EXPOSE 8000
ENTRYPOINT ["./startup.sh"]
