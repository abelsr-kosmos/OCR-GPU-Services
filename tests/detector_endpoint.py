import requests
import time
import os
import json
import concurrent.futures
from typing import Tuple, Optional, List, Dict, Any

def send_detector_request(request_id: int, file_path: str, url: str) -> Tuple[int, float, int, Optional[str]]:
    """
    Sends a POST request with an image file to the specified document detection URL
    and returns the request ID, duration, status code, and any error message.

    Args:
        request_id (int): An identifier for the request.
        file_path (str): The path to the image file.
        url (str): The URL of the detector endpoint.

    Returns:
        Tuple[int, float, int, Optional[str]]: (request_id, duration, status_code, error_message)
                                                status_code is -1 if the request failed before sending.
                                                error_message is None if successful.
    """
    headers = {
        'accept': 'application/json',
        # 'Content-Type: multipart/form-data' is handled by requests when using 'files'
    }
    start_time = time.time()
    status_code = -1
    error_message = None
    response = None # Initialize response to None
    # Suppress individual request prints for cleaner summary output
    # print_prefix = f"  [Conc {request_id // 1000}, Req {request_id % 1000}]" # Example prefix if needed

    try:
        # Open the file in binary read mode for each request
        with open(file_path, 'rb') as f:
            # Determine content type based on file extension (basic example)
            # Defaulting to image/jpeg as per curl example
            file_basename = os.path.basename(file_path)
            _, ext = os.path.splitext(file_basename)
            content_type = 'image/jpeg' # Default based on curl
            if ext.lower() == '.png':
                content_type = 'image/png'
            elif ext.lower() in ['.jpg', '.jpeg']:
                content_type = 'image/jpeg'
            # Add more types if the endpoint supports them

            files = {
                'file': (file_basename, f, content_type)
            }

            # Send the POST request
            # Increased timeout for potentially slower concurrent runs or complex detection
            response = requests.post(url, headers=headers, files=files, timeout=180)
            status_code = response.status_code

            # Raise an HTTPError for bad responses (4xx or 5xx) to be caught below
            response.raise_for_status()

            # Attempt to parse JSON to ensure valid response format
            try:
                response.json()
            except requests.exceptions.JSONDecodeError as json_err:
                # Treat JSON decode error as a specific type of failure if needed
                # For now, we still consider the HTTP request successful if status is 2xx
                # print(f"{print_prefix} Warning: Received non-JSON response (Status: {status_code}): {response.text[:100]}...")
                pass # Keep it silent for summary

    except requests.exceptions.HTTPError as http_err:
        # Include response text in the error message if available
        response_text = response.text[:200] if response else "N/A"
        error_message = f"HTTP Error: {http_err} - Response: {response_text}"
    except requests.exceptions.ConnectionError as conn_err:
        error_message = f"Connection Error: {conn_err}"
    except requests.exceptions.Timeout as timeout_err:
        error_message = f"Timeout Error: {timeout_err}"
    except requests.exceptions.RequestException as req_err:
        error_message = f"Request Exception: {req_err}"
    except FileNotFoundError:
        # Should be caught before calling this function, but included for safety
        error_message = f"File not found: {file_path}"
        status_code = -2 # Specific code for file not found within the function
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
    finally:
        end_time = time.time()
        duration = end_time - start_time

    # Suppress individual request prints
    # if error_message:
    #     print(f"{print_prefix} Failed: {error_message} (Duration: {duration:.4f}s)")
    # else:
    #     print(f"{print_prefix} Completed: Status {status_code} (Duration: {duration:.4f}s)")

    return request_id, duration, status_code, error_message


if __name__ == "__main__":
    # --- Configuration ---
    # URL for the document detection endpoint
    detector_url = 'https://7g6zdiwbhb412l-8000.proxy.runpod.net/detect/doc-detection/?hide_result=false'
    # Use a default image or allow override via environment variable
    # Using the image from the curl example
    image_file = 'CFE-Inferencia.jpg'
    # Number of requests per concurrency level (adjust if needed)
    requests_per_level = 20
    # Concurrency levels to test
    concurrency_levels = [1, 4, 8] # Test from 1 to 8 concurrent requests
    # --- End Configuration ---

    # Check if the file exists before starting
    if not os.path.exists(image_file):
        print(f"Error: Test image file not found at '{image_file}'")
        print("Please ensure the image file exists or set the DETECTOR_TEST_IMAGE environment variable.")
        exit(1) # Exit if file not found

    print(f"Starting Document Detector endpoint concurrency test for levels {min(concurrency_levels)} to {max(concurrency_levels)}")
    print(f"Target URL: {detector_url}")
    print(f"Image file: {image_file}")
    print(f"Requests per level: {requests_per_level}\n")

    summary_results: List[Dict[str, Any]] = []

    for num_concurrent_requests in concurrency_levels:
        print(f"--- Testing Concurrency Level: {num_concurrent_requests} ---")
        results = []
        total_start_time = time.time()

        # Use ThreadPoolExecutor for I/O-bound tasks (network requests)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            # Submit tasks
            # Assign unique IDs across all tests if needed, or just 0 to N-1 for this level
            futures = [
                executor.submit(send_detector_request, i, image_file, detector_url)
                for i in range(requests_per_level) # Send N requests for this level
            ]

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    # Catch exceptions raised *during* the future's execution if not caught inside send_detector_request
                    print(f'  A request generated an unhandled exception: {exc}')
                    # Append a placeholder or specific error result if needed
                    results.append((-1, 0.0, -99, str(exc))) # Example error placeholder

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        # --- Analysis for this level ---
        successful_requests = [r for r in results if r[3] is None and 200 <= r[2] < 300]
        failed_requests = [r for r in results if r[3] is not None or not (200 <= r[2] < 300)]

        level_summary = {
            "concurrency": num_concurrent_requests,
            "total_time": total_duration,
            "successful": len(successful_requests),
            "failed": len(failed_requests),
            "avg_duration": 0.0,
            "min_duration": 0.0,
            "max_duration": 0.0,
        }

        if successful_requests:
            total_success_duration = sum(r[1] for r in successful_requests)
            level_summary["avg_duration"] = total_success_duration / len(successful_requests)
            level_summary["max_duration"] = max(r[1] for r in successful_requests)
            level_summary["min_duration"] = min(r[1] for r in successful_requests)

        summary_results.append(level_summary)

        print(f"  Level {num_concurrent_requests} completed in {total_duration:.4f}s")
        print(f"  Successful: {level_summary['successful']}, Failed: {level_summary['failed']}")
        if level_summary['successful'] > 0:
            print(f"  Avg Duration (Successful): {level_summary['avg_duration']:.4f}s")
            print(f"  Min Duration (Successful): {level_summary['min_duration']:.4f}s")
            print(f"  Max Duration (Successful): {level_summary['max_duration']:.4f}s")
        if level_summary['failed'] > 0:
             # Optionally print details of failed requests for this level
             # print("  Failed request details:")
             # for req_id, duration, status, err in failed_requests:
             #      print(f"    Req {req_id}: Status {status}, Duration {duration:.4f}s, Error: {err}")
             pass # Keep summary clean
        print("-" * 30 + "\n") # Separator


    # --- Final Summary Table ---
    print("\n--- Overall Concurrency Test Summary ---")
    print(f"{'Concurrency':<12} | {'Total Time':<12} | {'Successful':<10} | {'Failed':<8} | {'Avg Time':<12} | {'Min Time':<12} | {'Max Time':<12}")
    print("-" * 90)

    for summary in summary_results:
        print(f"{summary['concurrency']:<12} | {summary['total_time']:<12.4f} | {summary['successful']:<10} | {summary['failed']:<8} | {summary['avg_duration']:<12.4f} | {summary['min_duration']:<12.4f} | {summary['max_duration']:<12.4f}")
