import requests
import time
import os
import json
import csv
import argparse
import concurrent.futures
from typing import Tuple, Optional, List, Dict, Any

def send_request(request_id: int, file_path: str, url: str, endpoint_type: str) -> Tuple[int, float, int, Optional[str], str]:
    """
    Sends a POST request with an image file to the specified URL endpoint
    and returns the request ID, duration, status code, any error message, and endpoint type.

    Args:
        request_id (int): An identifier for the request.
        file_path (str): The path to the image file.
        url (str): The URL of the endpoint.
        endpoint_type (str): The type of endpoint being tested (signature, ocr, detector, etc.)

    Returns:
        Tuple[int, float, int, Optional[str], str]: (request_id, duration, status_code, error_message, endpoint_type)
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

    try:
        # Open the file in binary read mode for each request
        with open(file_path, 'rb') as f:
            # Determine file type based on extension for Content-Type
            file_basename = os.path.basename(file_path)
            _, ext = os.path.splitext(file_path)
            content_type = 'application/octet-stream' # Default fallback
            if ext.lower() == '.png':
                content_type = 'image/png'
            elif ext.lower() in ['.jpg', '.jpeg']:
                content_type = 'image/jpeg'
            # Add more types if the endpoint supports them

            files = {
                'file': (file_basename, f, content_type)
            }

            # Send the POST request
            # Using a timeout similar to other endpoints, adjust if needed
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

    return request_id, duration, status_code, error_message, endpoint_type


def run_concurrency_test(endpoint_type: str, endpoint_url: str, image_file: str, 
                         concurrency_levels: List[int], requests_per_level: int) -> List[Dict[str, Any]]:
    """
    Run a concurrency test for a specific endpoint.
    
    Args:
        endpoint_type: Type of endpoint (signature, ocr, detector, etc.)
        endpoint_url: URL of the endpoint
        image_file: Path to the image file
        concurrency_levels: List of concurrency levels to test
        requests_per_level: Number of requests to send per concurrency level
        
    Returns:
        List of dictionaries containing results for each concurrency level
    """
    # Check if the file exists before starting
    if not os.path.exists(image_file):
        print(f"Error: Test image file not found at '{image_file}'")
        print(f"Please ensure the image file exists")
        return []

    print(f"Starting {endpoint_type.upper()} endpoint concurrency test for levels {min(concurrency_levels)} to {max(concurrency_levels)}")
    print(f"Target URL: {endpoint_url}")
    print(f"Image file: {image_file}")
    print(f"Requests per level: {requests_per_level}\n")

    summary_results = []

    for num_concurrent_requests in concurrency_levels:
        print(f"--- Testing Concurrency Level: {num_concurrent_requests} ---")
        results = []
        total_start_time = time.time()

        # Use ThreadPoolExecutor for I/O-bound tasks (network requests)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            # Submit tasks
            futures = [
                executor.submit(send_request, i, image_file, endpoint_url, endpoint_type)
                for i in range(requests_per_level) # Send N requests for this level
            ]

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    # Catch exceptions raised *during* the future's execution if not caught inside send_request
                    print(f'  A request generated an unhandled exception: {exc}')
                    # Append a placeholder or specific error result if needed
                    results.append((-1, 0.0, -99, str(exc), endpoint_type)) # Example error placeholder

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        # --- Analysis for this level ---
        successful_requests = [r for r in results if r[3] is None and 200 <= r[2] < 300]
        failed_requests = [r for r in results if r[3] is not None or not (200 <= r[2] < 300)]

        level_summary = {
            "endpoint": endpoint_type,
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
            # for req_id, duration, status, err, _ in failed_requests:
            #      print(f"    Req {req_id}: Status {status}, Duration {duration:.4f}s, Error: {err}")
            pass # Keep summary clean
        print("-" * 30 + "\n") # Separator

    return summary_results


def print_summary_table(all_results: List[Dict[str, Any]]):
    """Print a summary table of all test results"""
    if not all_results:
        print("\nNo results to display.")
        return
        
    # Group results by endpoint
    endpoint_results = {}
    for result in all_results:
        endpoint = result["endpoint"]
        if endpoint not in endpoint_results:
            endpoint_results[endpoint] = []
        endpoint_results[endpoint].append(result)
    
    for endpoint, results in endpoint_results.items():
        print(f"\n--- Overall Concurrency Test Summary ({endpoint.upper()} Endpoint) ---")
        header_str = f"{'Concurrency':<12} | {'Total Time':<12} | {'Successful':<10} | {'Failed':<8} | {'Avg Time':<12} | {'Min Time':<12} | {'Max Time':<12}"
        print(header_str)
        print("-" * len(header_str))
        
        for summary in results:
            print(f"{summary['concurrency']:<12} | {summary['total_time']:<12.4f} | {summary['successful']:<10} | {summary['failed']:<8} | {summary['avg_duration']:<12.4f} | {summary['min_duration']:<12.4f} | {summary['max_duration']:<12.4f}")


def save_results_to_csv(all_results: List[Dict[str, Any]], filename: str):
    """Save all test results to a CSV file"""
    if not all_results:
        print(f"\nNo results to save to CSV file '{filename}'.")
        return
        
    try:
        # Define fieldnames based on the keys of the first summary dictionary
        fieldnames = list(all_results[0].keys())
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header row
            writer.writeheader()
            
            # Write data rows
            writer.writerows(all_results)
            
        print(f"\nSummary results successfully saved to {filename}")
    except IOError as e:
        print(f"\nError: Could not write summary results to CSV file '{filename}'. Reason: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred while writing to CSV: {e}")


def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Unified API endpoint concurrency testing tool")
    parser.add_argument("--endpoints", nargs="+", choices=["signature", "ocr", "detector", "qr", "render", "all"], 
                        default=["all"], help="Endpoints to test")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8], 
                        help="Concurrency levels to test")
    parser.add_argument("--requests", type=int, default=20, 
                        help="Number of requests per concurrency level")
    parser.add_argument("--output", type=str, default="endpoint_concurrency_results.csv", 
                        help="CSV output filename")
    args = parser.parse_args()
    
    # Configuration for different endpoints
    endpoint_config = {
        "signature": {
            "url": "https://7ye8qosdl3igmb-8000.proxy.runpod.net/tools/signature?return_img=false",
            "image": "CFE-Inferencia.jpg"
        },
        "ocr": {
            "url": "https://7ye8qosdl3igmb-8000.proxy.runpod.net/tools/paddle-ocr",
            "image": "RECIBO_TELMEX.jpg"
        },
        "detector": {
            "url": "https://7ye8qosdl3igmb-8000.proxy.runpod.net/detect/doc-detection/?hide_result=false",
            "image": "CFE-Inferencia.jpg"
        },
        "qr": {
            "url": "https://7ye8qosdl3igmb-8000.proxy.runpod.net/tools/qr",
            "image": "CFE-Inferencia.jpg"
        },
        "render": {
            "url": "https://7ye8qosdl3igmb-8000.proxy.runpod.net/tools/doctr?operation=render",
            "image": "RECIBO_TELMEX.jpg"
        }
    }
    
    # Determine which endpoints to test
    endpoints_to_test = []
    if "all" in args.endpoints:
        endpoints_to_test = list(endpoint_config.keys())
    else:
        endpoints_to_test = args.endpoints
    
    # Run the tests for each endpoint
    all_results = []
    
    for endpoint in endpoints_to_test:
        if endpoint in endpoint_config:
            config = endpoint_config[endpoint]
            results = run_concurrency_test(
                endpoint_type=endpoint,
                endpoint_url=config["url"],
                image_file=config["image"],
                concurrency_levels=args.concurrency,
                requests_per_level=args.requests
            )
            all_results.extend(results)
        else:
            print(f"Warning: Unknown endpoint '{endpoint}'. Skipping.")
    
    # Print summary and save to CSV
    if all_results:
        print_summary_table(all_results)
        save_results_to_csv(all_results, args.output)
    else:
        print("No test results were generated. Please check your configuration and try again.")


if __name__ == "__main__":
    main() 