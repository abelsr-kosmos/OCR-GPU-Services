# 🔍 Kosmos OCR General GPU Services 🚀

## 📋 Overview
This project provides a set of GPU-accelerated OCR (Optical Character Recognition) and document processing services through a FastAPI application. It offers various tools for document classification, text extraction, QR code detection, signature verification, and document rendering.

## ✨ Features
- **Document Classification** 📄: Classify documents based on their content and structure
- **OCR Processing** 🔤: Extract text from images using PaddleOCR and DocTR
- **QR Code Detection** 📱: Identify and decode QR codes in documents
- **Signature Verification** ✍️: Detect and verify signatures in documents
- **Document Detection** 🔎: Identify document boundaries and types
- **Document Rendering** 🖼️: Generate visual representations of documents

## 🌐 API Endpoints

### Classification
- `POST /api/v1/classify/`: Classify multiple documents with reference information

### Tools
- `POST /api/v1/tools/paddle-ocr`: Process documents using PaddleOCR
- `POST /api/v1/tools/doctr`: Process documents using DocTR
- `POST /api/v1/tools/classify`: Classify a single document
- `POST /api/v1/tools/qr`: Detect QR codes in documents
- `POST /api/v1/tools/signature`: Verify signatures in documents
- `POST /api/v1/tools/doc-detector`: Detect document boundaries
- `POST /api/v1/tools/render`: Render documents

## 🔧 Technical Details
- Built with FastAPI for high-performance API endpoints
- Utilizes GPU acceleration with Paddle for OCR processing
- Implements a service-oriented architecture with dependency injection
- Processes documents as entities through a domain-driven design approach

## 📂 Project Structure

```
OCR-GPU-Services/
├── app/             # Main application source code
│   ├── api/         # API endpoint definitions
│   ├── core/        # Core configuration and settings
│   ├── domain/      # Domain models and entities
│   ├── services/    # Business logic and services
│   └── main.py      # FastAPI application entry point
├── docker/          # Docker related files (Dockerfile, etc.)
├── notebooks/       # Jupyter notebooks for experimentation
├── .venv/           # Virtual environment (usually excluded from Git)
├── .git/            # Git repository files
├── README.md        # Project description and instructions
└── requirements.txt # Python dependencies
```

## 🐳 Docker Instructions

### Prerequisites
- Docker installed on your system.
- For GPU support: NVIDIA drivers and NVIDIA Container Toolkit installed on the host machine.

### Building the Image

**GPU Version:**
```bash
docker build -t ocr-gpu-services -f docker/Dockerfile .
```

**CPU Version:**
```bash
# Note: Ensure your requirements.txt is suitable for CPU or create a separate requirements-cpu.txt
docker build -t ocr-cpu-services -f docker/Dockerfile.cpu .
```

### Running the Container

**GPU Version:**
```bash
# Requires NVIDIA drivers and NVIDIA Container Toolkit on the host
docker run --gpus all -p 8000:8000 ocr-gpu-services
```

**CPU Version:**
```bash
docker run -p 8000:8000 ocr-cpu-services
```
