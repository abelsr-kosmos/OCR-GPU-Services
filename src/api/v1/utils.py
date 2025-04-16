"""Utilities for API v1 routes"""
import asyncio
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Thread pool for CPU-bound tasks (shared across routers)
executor = ThreadPoolExecutor(max_workers=4)

async def run_in_threadpool(func, *args):
    """Run CPU-bound functions in a thread pool to avoid blocking the event loop"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)

async def process_image(image_data: bytes):
    """Convert image bytes to OpenCV format efficiently"""
    nparr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

async def run_concurrent_tasks(tasks_dict):
    """
    Run multiple tasks concurrently and return results with their keys
    
    Args:
        tasks_dict: Dictionary mapping keys to coroutines
        
    Returns:
        Dictionary mapping keys to task results
    """
    if not tasks_dict:
        return {}
        
    # Create tasks list and matching keys list
    keys = list(tasks_dict.keys())
    tasks = [tasks_dict[key] for key in keys]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Map results back to their keys
    return {keys[i]: results[i] for i in range(len(keys))} 