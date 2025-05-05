import io
import time
import traceback
from typing import Literal, Any, List, Dict, Tuple, Union

import gc
import torch
import threading
from PIL import Image
from loguru import logger
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from app.schemas.schemas import OCROut, OCRPage, OCRBlock, OCRLine, OCRWord


class DocTRRunner:
    _instance = None
    _model = None
    _lock = threading.Lock()
    
    def __new__(cls):
        # Implement a singleton pattern to ensure we only have one model in memory
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DocTRRunner, cls).__new__(cls)
        return cls._instance
        
    def __init__(self):
        # Only initialize the model once
        if DocTRRunner._model is None:
            with self._lock:
                if DocTRRunner._model is None:
                    logger.info("Initializing DocTR model...")
                    t_start = time.time()
                    DocTRRunner._model = self._initialize_model()
                    logger.info(f"DocTR model initialization took {time.time() - t_start:.4f} seconds")
        self.model = DocTRRunner._model
        
    def _initialize_model(self):
        # Initialize the model with optimized settings
        t_start = time.time()
        logger.info("Loading DocTR model...")
        import torch
        try:
            # Try to determine if we can use a lighter model based on available memory
            use_light_model = False
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_mem < 4.0:  # Less than 4GB memory available
                    use_light_model = True
                    logger.info(f"Limited GPU memory ({gpu_mem:.1f}GB), using lightweight model")
            
            # Choose model architecture based on available resources
            if use_light_model:
                model = ocr_predictor(pretrained=True, det_arch='db_resnet34', reco_arch='crnn_vgg16_bn')
                logger.info("Using lightweight model architecture")
            else:
                model = ocr_predictor(pretrained=True)
                logger.info("Using standard model architecture")
                
            logger.info(f"Base model loaded in {time.time() - t_start:.4f} seconds")
            
            if torch.cuda.is_available():
                # Record GPU info
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
                
                # Check for TensorRT availability
                tensorrt_available = False
                try:
                    import tensorrt
                    import torch_tensorrt
                    tensorrt_available = True
                    logger.info(f"TensorRT {tensorrt.__version__} is available")
                except ImportError:
                    logger.info("TensorRT not available, using standard PyTorch")
                
                # Use TorchScript for faster inference if compatible
                try:
                    # Move model to GPU
                    t_gpu = time.time()
                    model.to("cuda").eval()
                    logger.info(f"Model moved to GPU in {time.time() - t_gpu:.4f} seconds")
                    
                    # Apply quantization and optimization
                    t_optimize = time.time()
                    
                    # Try INT8 quantization if available
                    try:
                        # For PyTorch 1.10+
                        if hasattr(torch.quantization, 'quantize_dynamic'):
                            # Quantize the detection model
                            if hasattr(model.det_predictor, 'model'):
                                # Try to quantize to INT8
                                model.det_predictor.model = torch.quantization.quantize_dynamic(
                                    model.det_predictor.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
                                )
                                logger.info("Detection model quantized to INT8")
                        else:
                            # Fallback to half precision
                            model.det_predictor.model = model.det_predictor.model.half()
                            model.reco_predictor.model = model.reco_predictor.model.half()
                            logger.info("Models converted to FP16 precision")
                    except Exception as qe:
                        # Fallback to half precision
                        logger.warning(f"INT8 quantization failed: {qe}, falling back to FP16")
                        model.det_predictor.model = model.det_predictor.model.half()
                        model.reco_predictor.model = model.reco_predictor.model.half()
                    
                    # Apply TensorRT if available for detection model (the most computationally intensive part)
                    if tensorrt_available and not use_light_model:
                        try:
                            # Try to convert detection model to TensorRT
                            trt_model = torch_tensorrt.compile(
                                model.det_predictor.model,
                                inputs=[torch_tensorrt.Input(
                                    min_shape=[1, 3, 480, 480],
                                    opt_shape=[1, 3, 640, 640],
                                    max_shape=[1, 3, 1280, 1280],
                                    dtype=torch.half
                                )],
                                enabled_precisions={torch.half},  # Use FP16
                                workspace_size=1 << 28,  # 256MB workspace
                            )
                            model.det_predictor.model = trt_model
                            logger.info("Detection model accelerated with TensorRT")
                        except Exception as te:
                            logger.warning(f"TensorRT acceleration failed: {te}")
                    
                    # Enable cuDNN benchmarking for optimal performance
                    torch.backends.cudnn.benchmark = True
                    
                    logger.info(f"Model optimized in {time.time() - t_optimize:.4f} seconds")
                    
                    # Set lower precision for inference to save memory
                    with torch.cuda.amp.autocast():
                        # Warm up the model with a dummy input to precompile CUDA kernels
                        t_warmup = time.time()
                        dummy_input = torch.zeros((1, 3, 640, 640), device="cuda", dtype=torch.float16)
                        # Dummy forward pass to initialize CUDA kernels
                        _ = model.det_predictor.model(dummy_input)
                        logger.info(f"Model warmup completed in {time.time() - t_warmup:.4f} seconds")
                        
                    logger.info(f"DocTR model initialized on GPU with optimized precision")
                except Exception as e:
                    # Fallback to standard GPU without mixed precision
                    logger.warning(f"Failed to initialize with optimized settings: {e}")
                    logger.info("Falling back to standard precision on GPU")
                    torch.cuda.empty_cache()  # Clear any failed allocations
                    model = ocr_predictor(pretrained=True)  # Reload model
                    model.to("cuda").eval()
                    logger.info("DocTR model initialized on GPU with standard precision")
            else:
                logger.info("No GPU detected, using CPU")
                # Try to optimize for CPU
                model.to("cpu").eval()
                
                # Try to use MKL and other CPU optimizations
                try:
                    import torch.backends.mkldnn
                    if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.enabled:
                        logger.info("MKL-DNN acceleration enabled for CPU")
                    
                    # Set number of threads for CPU processing
                    import multiprocessing
                    num_cores = multiprocessing.cpu_count()
                    torch.set_num_threads(num_cores)
                    logger.info(f"Using {num_cores} CPU threads for inference")
                except Exception as ce:
                    logger.warning(f"CPU optimization failed: {ce}")
                
                logger.info("DocTR model initialized on CPU")
                
            return model
            
        except Exception as e:
            logger.error(f"Trace: {traceback.format_exc()}")
            logger.error(f"Error initializing DocTR model: {str(e)}")
            # Last resort - try CPU with minimal settings
            try:
                logger.info("Attempting fallback to basic CPU model")
                model = ocr_predictor(pretrained=True, det_arch='db_resnet34', reco_arch='crnn_vgg16_bn')
                model.to("cpu").eval()
                return model
            except Exception as e2:
                logger.critical(f"Critical error initializing model: {str(e2)}")
                raise RuntimeError(f"Failed to initialize DocTR model: {str(e)}, {str(e2)}")

    def _preprocess_image(self, image_bytes: bytes) -> DocumentFile:
        """Optimize image preprocessing to reduce overhead"""
        try:
            # Use PIL's efficient loading and convert to RGB directly
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            logger.debug(f"Image size: {image.size}")
            
            # Optimize image size if needed (DocTR works best with reasonable resolutions)
            w, h = image.size
            max_dim = 2000
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                new_size = (int(w * scale), int(h * scale))
                image = image.resize(new_size, Image.LANCZOS)
                image = image.convert('RGB')
                logger.debug(f"Resized image from {w}x{h} to {new_size[0]}x{new_size[1]}")
            
            # Convert to bytes
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='JPEG', optimize=True, quality=95)
            image_bytes = image_bytes.getvalue()
            logger.debug(f"Image now on bytes")
            
            # Create DocumentFile correctly
            doc = DocumentFile.from_images([image_bytes])
            return doc
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            # Fallback to original approach
            try:
                return DocumentFile.from_images(image_bytes)
            except:
                # Last resort fallback
                return DocumentFile.from_images([image_bytes])

    def render(
        self, file: bytes, file_type: Literal["image", "pdf"]
    ) -> List[Dict]:
        start_time = time.time()
        
        try:
            # Use optimized preprocessing for images
            t_preprocess = time.time()
            if file_type in ["jpg", "jpeg", "png"]:
                doc = self._preprocess_image(file)
            elif file_type == "pdf":
                doc = DocumentFile.from_pdf(file)
            else:
                raise ValueError("Invalid file type")
            logger.debug(f"Preprocessing completed in {time.time() - t_preprocess:.4f} seconds")

            # Record model inference time separately
            t_inference = time.time()
            
            try:
                # Use mixed precision for inference with non-blocking CUDA operations
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            # Use CUDA streams for overlapping operations
                            with torch.cuda.stream(torch.cuda.Stream()):
                                result = self.model(doc).render()
                        else:
                            result = self.model(doc).render()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Handle OOM error
                    logger.warning("GPU out of memory, clearing cache and retrying")
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Retry with reduced precision
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        with torch.no_grad():
                            result = self.model(doc).render()
                else:
                    # Handle other runtime errors
                    logger.error(f"Runtime error during inference: {str(e)}")
                    # Try to continue with CPU if GPU failed
                    if torch.cuda.is_available():
                        logger.info("Attempting to fall back to CPU")
                        # Move model to CPU and try again
                        try:
                            self.model.to("cpu")
                            with torch.no_grad():
                                result = self.model(doc).render()
                            # Move back to GPU for next time
                            self.model.to("cuda")
                        except Exception as cpu_e:
                            logger.error(f"CPU fallback also failed: {str(cpu_e)}")
                            raise
                    else:
                        raise
            
            logger.info(f"Model inference time: {time.time() - t_inference:.4f} seconds")
                
            # Clear GPU cache after processing asynchronously
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Failed to clear CUDA cache: {str(e)}")
                
            end_time = time.time()
            logger.info(f"Time taken to render document: {end_time - start_time} seconds")
            
            return result
        except Exception as e:
            logger.error(f"Error rendering document: {str(e)}")
            # Try to provide a meaningful error response
            if "RuntimeError: CUDA error" in str(e):
                raise ValueError("GPU error occurred. Try again or contact support.")
            elif "out of memory" in str(e).lower():
                raise ValueError("System is low on memory. Try with a smaller document.")
            else:
                raise ValueError(f"Error rendering document: {e}")

    def ocr(
        self, file: bytes, file_type: Literal["image", "pdf"]
    ) -> List[Dict]:
        start_time = time.time()
        
        try:
            # Use optimized preprocessing for images
            t_preprocess = time.time()
            if file_type in ["jpg", "jpeg", "png"]:
                doc = self._preprocess_image(file)
            elif file_type == "pdf":
                doc = DocumentFile.from_pdf(file)
            else:
                raise ValueError("Invalid file type")
            logger.debug(f"Preprocessing completed in {time.time() - t_preprocess:.4f} seconds")

            # Record model inference time separately
            t_inference = time.time()
            
            try:
                # Use mixed precision for inference
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            # Use CUDA streams for overlapping operations
                            with torch.cuda.stream(torch.cuda.Stream()):
                                result = self.model(doc)
                        else:
                            result = self.model(doc)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Handle OOM error
                    logger.warning("GPU out of memory, clearing cache and retrying")
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Retry with reduced precision
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        with torch.no_grad():
                            result = self.model(doc)
                else:
                    # Handle other runtime errors
                    logger.error(f"Runtime error during inference: {str(e)}")
                    # Try to continue with CPU if GPU failed
                    if torch.cuda.is_available():
                        logger.info("Attempting to fall back to CPU")
                        # Move model to CPU and try again
                        try:
                            self.model.to("cpu")
                            with torch.no_grad():
                                result = self.model(doc)
                            # Move back to GPU for next time
                            self.model.to("cuda")
                        except Exception as cpu_e:
                            logger.error(f"CPU fallback also failed: {str(cpu_e)}")
                            raise
                    else:
                        raise
            
            logger.info(f"Model inference time: {time.time() - t_inference:.4f} seconds")
                
            # Clear GPU cache after processing
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Failed to clear CUDA cache: {str(e)}")

            # Improve response generation efficiency
            t_response = time.time()
            
            # Use more efficient list comprehension and avoid redundant calculations
            # Pre-compute some values for better performance
            results = []
            for i, page in enumerate(result.pages):
                # Create block data using comprehension for better performance
                blocks = []
                for block in page.blocks:
                    lines = []
                    for line in block.lines:
                        # Optimize word creation
                        words = [
                            OCRWord(
                                value=word.value,
                                geometry=self.resolve_geometry(word.geometry),
                                objectness_score=round(word.objectness_score, 2),
                                confidence=round(word.confidence, 2),
                                crop_orientation=word.crop_orientation,
                            )
                            for word in line.words
                        ]
                        
                        lines.append(OCRLine(
                            geometry=self.resolve_geometry(line.geometry),
                            objectness_score=round(line.objectness_score, 2),
                            words=words,
                        ))
                        
                    blocks.append(OCRBlock(
                        geometry=self.resolve_geometry(block.geometry),
                        objectness_score=round(block.objectness_score, 2),
                        lines=lines,
                    ))
                
                # Create the page entity
                results.append(OCROut(
                    name=str(i),
                    orientation=page.orientation,
                    language=page.language,
                    dimensions=page.dimensions,
                    items=[OCRPage(blocks=blocks)]
                ))
            
            logger.info(f"Response generation time: {time.time() - t_response:.4f} seconds")
            logger.info(f"Total OCR time: {time.time() - start_time:.4f} seconds")
            
            return results
        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}")
            raise ValueError(f"Error in OCR processing: {e}")

    def resolve_geometry(
        self,
        geom: Any,
    ) -> (
        Union[Tuple[float, float, float, float],  Tuple[float, float, float, float, float, float, float, float]]
    ):
        if len(geom) == 4:
            return (*geom[0], *geom[1], *geom[2], *geom[3])
        return (*geom[0], *geom[1])
