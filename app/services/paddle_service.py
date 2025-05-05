from typing import List, Union
import time
import numpy as np
from PIL import Image
import cv2
from app.infrastructure.ml.paddle_runner import PaddleOCRRunner
from loguru import logger
from functools import lru_cache
import threading
import concurrent.futures
import paddle

# Thread-local storage for reusable OpenCV objects
_thread_local = threading.local()

class PaddleService:
    def __init__(self) -> None:
        t_start = time.time()
        self._paddle_runner = PaddleOCRRunner()
        # Set up CUDA stream for asynchronous GPU operations if CUDA is available
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_cuda:
            logger.info(f"CUDA is available for CV operations with {cv2.cuda.getCudaEnabledDeviceCount()} devices")
        logger.info(f"PaddleService initialized in {time.time() - t_start:.4f} seconds")

    def _get_clahe(self):
        """Get thread-local CLAHE object to avoid recreation"""
        if not hasattr(_thread_local, 'clahe'):
            _thread_local.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return _thread_local.clahe

    @lru_cache(maxsize=32)
    def _calculate_resize_dimensions(self, width: int, height: int, max_dim: int = 2000, min_dim: int = 800) -> tuple:
        """Calculate resize dimensions and cache the results"""
        need_resize = False
        scale = 1.0
        
        # Resize large images
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            need_resize = True
        # Upsample very small images to improve text recognition
        elif max(height, width) < min_dim:
            scale = min_dim / max(height, width)
            need_resize = True
            
        if need_resize:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return new_width, new_height, True, scale
        
        return width, height, False, scale

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for better OCR performance with GPU acceleration
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed numpy array
        """
        t_start = time.time()
        
        # Convert PIL image to numpy array with more efficient path
        if hasattr(image, 'mode') and image.mode == 'RGB':
            # Faster direct conversion to numpy array
            image_np = np.asarray(image, dtype=np.uint8)
        else:
            # Convert to RGB first if needed
            image_np = np.array(image.convert('RGB'), dtype=np.uint8)
        
        # Check image dimensions for resizing
        height, width = image_np.shape[:2]
        new_width, new_height, need_resize, scale = self._calculate_resize_dimensions(width, height)
            
        # Only log resizing if actually performed
        if need_resize:
            # Use GPU resize if available
            if self.use_cuda and hasattr(cv2, 'cuda'):
                try:
                    gpu_img = cv2.cuda_GpuMat()
                    gpu_img.upload(image_np)
                    
                    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                    gpu_img = cv2.cuda.resize(gpu_img, (new_width, new_height), interpolation=interpolation)
                    image_np = gpu_img.download()
                    logger.debug(f"Resized image on GPU from {width}x{height} to {new_width}x{new_height}")
                except Exception as e:
                    # Fallback to CPU resize
                    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                    image_np = cv2.resize(image_np, (new_width, new_height), interpolation=interpolation)
                    logger.debug(f"Resized image on CPU: {str(e)}")
            else:
                # CPU resize
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                image_np = cv2.resize(image_np, (new_width, new_height), interpolation=interpolation)
                logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        # Fast path for grayscale conversion
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            # GPU accelerated grayscale if possible
            if self.use_cuda and hasattr(cv2, 'cuda'):
                try:
                    gpu_img = cv2.cuda_GpuMat()
                    gpu_img.upload(image_np)
                    gray_gpu = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_RGB2GRAY)
                    gray = gray_gpu.download()
                except Exception:
                    # Fallback to CPU
                    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # Calculate edge density using a faster sampling approach
            # Only sample a portion of the image to speed up edge detection
            sample_factor = 0.5  # Use 50% of pixels for edge detection
            if min(gray.shape) > 500:
                sample_h, sample_w = int(gray.shape[0] * sample_factor), int(gray.shape[1] * sample_factor)
                gray_sample = cv2.resize(gray, (sample_w, sample_h), interpolation=cv2.INTER_AREA)
                edges = cv2.Canny(gray_sample, 100, 200)
                edge_density = edges.mean() / 255.0
            else:
                edges = cv2.Canny(gray, 100, 200)
                edge_density = edges.mean() / 255.0
            
            # Fast path for low-detail images (skip expensive binarization and denoising)
            if edge_density <= 0.1:
                # For images with low text density, use contrast-enhanced grayscale
                image_np = self._get_clahe().apply(gray)
                logger.info(f"Using contrast-enhanced grayscale for OCR (edge density: {edge_density:.2f})")
            else:
                # Apply adaptive thresholding with optimized parameters
                block_size = 11  # Size of pixel neighborhood for adaptive threshold
                c_value = 9      # Constant subtracted from mean
                
                # GPU accelerated binary threshold if possible
                if self.use_cuda and hasattr(cv2, 'cuda'):
                    try:
                        gpu_gray = cv2.cuda_GpuMat()
                        gpu_gray.upload(gray)
                        # CUDA doesn't have adaptiveThreshold, so we use an alternative approach
                        gpu_blur = cv2.cuda.GaussianBlur(gpu_gray, (block_size, block_size), 0)
                        blur = gpu_blur.download()
                        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                    except Exception:
                        # Fallback to CPU version
                        binary = cv2.adaptiveThreshold(
                            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            cv2.THRESH_BINARY, block_size, c_value
                        )
                else:
                    binary = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                        cv2.THRESH_BINARY, block_size, c_value
                    )
                
                # Skip denoising for high edge density images (text-heavy)
                if edge_density < 0.2:
                    # Fast NL means denoising with optimized parameters
                    # Reduce h parameter for faster processing
                    binary = cv2.fastNlMeansDenoising(binary, None, 7, 5, 19)
                
                image_np = binary
                logger.info(f"Using binarized image for OCR (edge density: {edge_density:.2f})")
        
        # Ensure image is in uint8 format (avoid unnecessary conversion)
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
        
        logger.info(f"Image preprocessing completed in {time.time() - t_start:.4f} seconds")
        return image_np

    def ocr(self, image: Image.Image) -> List[dict]:
        """
        Process an image with PaddleOCR with optimized performance
        
        Args:
            image: PIL Image to process
            
        Returns:
            List of OCR results
        """
        t_start = time.time()
        
        # Convert PIL image to numpy array with preprocessing
        image_np = self._preprocess_image(image)
        
        # Run OCR prediction
        t_ocr = time.time()
        result = self._paddle_runner.predict(image_np)
        logger.info(f"OCR prediction completed in {time.time() - t_ocr:.4f} seconds")
        
        logger.info(f"Total OCR processing completed in {time.time() - t_start:.4f} seconds")
        return result
    
    def batch_ocr(self, images: List[Image.Image]) -> List[List[dict]]:
        """
        Process multiple images with PaddleOCR in batch with parallel preprocessing
        
        This is more efficient than processing images one by one
        
        Args:
            images: List of PIL Images to process
            
        Returns:
            List of OCR results, one for each input image
        """
        t_start = time.time()
        logger.info(f"Starting batch OCR processing for {len(images)} images")
        
        # Early return for empty batch
        if not images:
            return []
        
        # Use parallel preprocessing if multiple images are provided
        if len(images) > 1:
            try:
                # Define preprocessing worker
                def preprocess_worker(img):
                    try:
                        return self._preprocess_image(img)
                    except Exception as e:
                        logger.error(f"Error preprocessing image: {str(e)}")
                        # Return original image as fallback
                        return np.array(img) if isinstance(img, Image.Image) else img
                
                # Use thread pool for I/O bound preprocessing operations
                t_preprocess = time.time()
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(images))) as executor:
                    image_nps = list(executor.map(preprocess_worker, images))
                logger.info(f"Parallel preprocessing completed in {time.time() - t_preprocess:.4f} seconds")
            except Exception as e:
                # Fallback to sequential preprocessing
                logger.warning(f"Parallel preprocessing failed: {str(e)}, falling back to sequential")
                t_preprocess = time.time()
                image_nps = [self._preprocess_image(img) for img in images]
                logger.info(f"Sequential preprocessing completed in {time.time() - t_preprocess:.4f} seconds")
        else:
            # Just one image - process normally
            t_preprocess = time.time()
            image_nps = [self._preprocess_image(images[0])]
            logger.info(f"Single image preprocessing completed in {time.time() - t_preprocess:.4f} seconds")
        
        # Process images in optimal batches
        # Dynamically adjust batch size based on image dimensions to optimize GPU memory usage
        results = []
        
        # Function to estimate optimal batch size based on image dimensions
        def estimate_batch_size(images):
            if not images:
                return 16
            
            # Calculate average image size
            avg_pixels = np.mean([img.shape[0] * img.shape[1] for img in images])
            
            # Adjust batch size based on image size
            if avg_pixels > 4000000:  # Very large images (>2000x2000)
                return 4
            elif avg_pixels > 1000000:  # Large images (>1000x1000)
                return 8
            else:  # Smaller images
                return 16
        
        # Get optimal batch size
        batch_size = estimate_batch_size(image_nps)
        logger.info(f"Using dynamic batch size of {batch_size} based on image dimensions")
        
        for i in range(0, len(image_nps), batch_size):
            t_batch = time.time()
            batch = image_nps[i:i+batch_size]
            
            try:
                # Process batch with error recovery
                batch_results = []
                for idx, img in enumerate(batch):
                    try:
                        result = self._paddle_runner.predict(img)
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing image {i+idx}: {str(e)}")
                        # Add empty result to maintain order
                        batch_results.append([])
                        continue
                        
                # Add batch results to overall results
                results.extend(batch_results)
                
                logger.info(f"Batch {i//batch_size + 1}/{(len(image_nps)+batch_size-1)//batch_size} processed in {time.time() - t_batch:.4f} seconds")
                
                # Clear GPU memory after each batch if using GPU
                if self._paddle_runner.use_gpu:
                    paddle.device.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                # Add empty results for this batch to maintain correct length
                results.extend([[] for _ in range(len(batch))])
                continue
        
        logger.info(f"Total batch OCR processing completed in {time.time() - t_start:.4f} seconds")
        return results
