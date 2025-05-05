from typing import List

import time
import paddle
import numpy as np
from loguru import logger
from paddleocr import PaddleOCR

from app.domain.entities import PageEntity, ItemEntity



class PaddleOCRRunner:
    def __init__(self):
        t_start = time.time()
        # Check if CUDA is available
        self.use_gpu = paddle.device.is_compiled_with_cuda()
        gpu_mem = 2000
        
        # Configure based on GPU availability
        if self.use_gpu:
            # Log GPU device information
            logger.info(f"Using GPU for PaddleOCR: {paddle.device.cuda.get_device_name(0)}")
            
            try:
                # Get GPU properties and allocate memory more aggressively
                gpu_info = paddle.device.cuda.get_device_properties(0)
                total_memory = gpu_info.total_memory / (1024 * 1024) 
                # Use 90% of available GPU memory for better utilization
                gpu_mem = int(total_memory * 0.9)
                logger.info(f"Allocating {gpu_mem}MB of GPU memory for PaddleOCR")
            except Exception as e:
                logger.warning(f"Failed to get GPU properties: {e}, using default memory allocation")
        
        # Additional optimizations for Paddle
        paddle.set_flags({
            'FLAGS_eager_delete_tensor_gb': 0.0,  # Enable garbage collection
            'FLAGS_memory_fraction_of_eager_deletion': 1.0,  # More aggressive GC
            'FLAGS_fast_eager_deletion_mode': 1,  # Use fast GC mode
            'FLAGS_allocator_strategy': 'auto_growth',  # Use auto growth for memory
        })
        
        # Enable graph compilation for faster inference (version-compatible approach)
        if hasattr(paddle, 'jit'):
            try:
                # Try simpler approach that's compatible with more Paddle versions
                paddle.jit.enable_to_static(True)
                logger.info("Static graph optimization enabled")
            except Exception as e:
                logger.warning(f"Failed to enable static graph optimization: {e}")
                
        # Initialize PaddleOCR with optimized settings
        ocr_args = {
            'use_angle_cls': True,
            'lang': "en",
            'det_lang': "ml",
            'show_log': False,
            'use_gpu': self.use_gpu,
            'gpu_mem': gpu_mem,
            'enable_mkldnn': not self.use_gpu,
            'cpu_threads': 8 if not self.use_gpu else 1,
            'use_tensorrt': False,
            'precision': 'trt_fp16' if self.use_gpu else 'fp16',
            'trt_min_subgraph_size': 5,
            'rec_batch_num': 64,
            'cls_batch_num': 64,
            'sr_batch_num': 64, 
            'max_batch_size': 64,
        }
            
        # Initialize OCR with the configured arguments
        self.ocr = PaddleOCR(**ocr_args)
        
        # Pre-warm the model with a dummy image to compile any lazy operations
        try:
            logger.info("Pre-warming model with dummy inference...")
            dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
            _ = self.ocr.ocr(dummy_image, cls=True)
        except Exception as e:
            logger.warning(f"Pre-warming failed: {e}, continuing anyway")
        
        logger.info(f"PaddleOCR initialization completed in {time.time() - t_start:.4f} seconds")

    def predict(self, image: np.ndarray) -> List[dict]:
        """Run OCR prediction with performance optimizations and caching"""
        t_start = time.time()
        
        try:
            # Ensure image has the right data type for best performance
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
                
            # Process based on device
            if self.use_gpu:
                # Set the appropriate GPU execution strategy
                paddle.set_device('gpu')
                
                # Enable memory optimization
                paddle.device.cuda.empty_cache()
                
                # Record GPU memory before processing
                t_process = time.time()
                with paddle.amp.auto_cast(enable=True):  # Enable mixed precision
                    result = self.ocr.ocr(image, cls=True)
                logger.info(f"OCR processing time: {time.time() - t_process:.4f} seconds")
            else:
                # Process on CPU
                t_process = time.time()
                result = self.ocr.ocr(image, cls=True)
                logger.info(f"OCR processing time: {time.time() - t_process:.4f} seconds")
            
            logger.info(f"Prediction completed in {time.time() - t_start:.4f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error during OCR prediction: {e}")
            raise RuntimeError(f"Error during OCR prediction: {e}")

    @staticmethod
    def process_result(
        ocr_results: List[dict], width: int, height: int
    ) -> PageEntity:
        """Process OCR results with vectorized operations for better performance"""
        t_start = time.time()
        pages = []
        texts = []  # Use list instead of string for better performance

        for result in ocr_results:
            json_output = []
            for page in result:
                for entry in page:
                    # More efficient coordinate processing
                    coordinates = entry[0]
                    coordinates = [list(map(int, coord)) for coord in coordinates]
                    text, confidence = entry[1]
                    json_output.append({
                        "Coordinates": coordinates,
                        "Text": text,
                        "Score": confidence,
                    })
                    texts.append(text)
            pages.append(json_output)

        # Create result entity directly without intermediate dict
        result = PageEntity(
            items=[
                ItemEntity(
                    coordinates=item["Coordinates"],
                    text=item["Text"],
                    score=item["Score"],
                )
                for item in pages[0] if pages
            ],
            text=texts,  # Join happens later when needed
            width=width,
            height=height,
        )
        
        logger.info(f"Results processed in {time.time() - t_start:.4f} seconds")
        return result
