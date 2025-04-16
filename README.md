# OCR Processing Service

A FastAPI service for processing and classifying documents with optional features like docTR, QR code detection, and signature detection.

## Project Structure

The project follows Clean Architecture principles:

```
src/
├── domain/                    # Core business logic
│   ├── entities/             # Business entities
│   └── repositories/         # Repository interfaces
├── application/              # Application services
│   └── services/            # Service implementations
└── infrastructure/           # External concerns
    ├── api/                 # API layer
    │   ├── main.py         # FastAPI application
    │   └── routes/         # API routes
    └── repositories/       # Repository implementations
```

## Features

- Clean Architecture implementation
- File processing and classification
- Optional docTR processing
- Optional QR code detection
- Optional signature detection
- Comprehensive logging
- Docker support
- CORS enabled

## Setup

### Local Development

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the service:
```bash
python -m src.infrastructure.api.main
```

The service will be available at `http://localhost:8000`

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t ocr-processing-service .
```

2. Run the container:
```bash
docker run -p 8000:8000 ocr-processing-service
```

The service will be available at `http://localhost:8000`

## Logging

The service includes comprehensive logging:
- Console output with human-readable format
- JSON-formatted logs in `app.log`
- Log rotation (10MB per file, 5 backup files)
- Request/response logging with timing information
- Error logging with stack traces

## API Endpoints

### 1. Classify and Process File
- **Endpoint**: `/api/v1/classify-n-process-file`
- **Method**: POST
- **Parameters**:
  - `file`: The file to process (required)
  - `doctr`: Boolean (optional, default: false) - Enable docTR processing
  - `qr`: Boolean (optional, default: false) - Enable QR code detection
  - `signature`: Boolean (optional, default: false) - Enable signature detection

### 2. Process File
- **Endpoint**: `/api/v1/process-file`
- **Method**: POST
- **Parameters**:
  - `file`: The file to process (required)
  - `qr`: Boolean (optional, default: false) - Enable QR code detection
  - `signature`: Boolean (optional, default: false) - Enable signature detection

## API Documentation

Once the service is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Example Usage

Using curl:
```bash
# Classify and process file with all features enabled
curl -X POST "http://localhost:8000/api/v1/classify-n-process-file?doctr=true&qr=true&signature=true" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_file.pdf"

# Process file with QR detection
curl -X POST "http://localhost:8000/api/v1/process-file?qr=true" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_file.pdf"
```

## Development

### Architecture

The project follows Clean Architecture principles:
1. **Domain Layer**: Contains business entities and repository interfaces
2. **Application Layer**: Contains use cases and business rules
3. **Infrastructure Layer**: Contains external concerns like API, database, etc.

### Logging Configuration

The service uses Python's built-in logging module with the following configuration:
- Console output for development
- JSON-formatted file logging for production
- Log rotation to prevent disk space issues
- Request/response timing information
- Error tracking with stack traces

### Docker Development

For development with Docker, you can mount your local directory to enable live code changes:
```bash
docker run -p 8000:8000 -v $(pwd):/app ocr-processing-service
``` 