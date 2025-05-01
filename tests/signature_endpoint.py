import requests
import time
import os
import json
import csv # Added import for CSV writing
import concurrent.futures
from typing import Tuple, Optional, List, Dict, Any

def send_signature_request(request_id: int, file_path: str, url: str) -> Tuple[int, float, int, Optional[str]]:
    """
    Sends a POST request with an image file to the specified signature detection URL
    and returns the request ID, duration, status code, and any error message.

    Args:
        request_id (int): An identifier for the request.
        file_path (str): The path to the image file.
        url (str): The URL of the signature endpoint (including query parameters).

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
            response = requests.post(url, headers=headers, files=files, timeout=120)
            status_code = response.status_code

            # Raise an HTTPError for bad responses (4xx or 5xx) to be caught below
            response.raise_for_status()

            # Attempt to parse JSON to ensure valid response format
            try:
                response.json()
            except requests.exceptions.JSONDecodeError as json_err:
                # Treat JSON decode error as a specific type of failure if needed
                # For now, we still consider the HTTP request successful if status is 2xx
                # print(f"Warning: Received non-JSON response (Status: {status_code}): {response.text[:100]}...")
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

    # Suppress individual request prints for cleaner summary output
    # if error_message:
    #     print(f"  Request {request_id}: Failed: {error_message} (Duration: {duration:.4f}s)")
    # else:
    #     print(f"  Request {request_id}: Completed: Status {status_code} (Duration: {duration:.4f}s)")

    return request_id, duration, status_code, error_message


if __name__ == "__main__":
    # --- Configuration ---
    # Default URL for the signature endpoint
    signature_url = 'https://10va4ptbkmb2bf-8000.proxy.runpod.net/tools/signature?return_img=false'
    # Use a default image or allow override via environment variable
    # Ensure this image actually contains signatures for meaningful testing
    image_file = 'CFE-Inferencia.jpg' # Example filename, replace with an actual test file
    # Number of requests per concurrency level
    requests_per_level = 20
    # Concurrency levels to test
    concurrency_levels = [1, 4, 8]
    # CSV output filename
    csv_filename = "signature_concurrency_summary.csv"
    # --- End Configuration ---

    # Check if the file exists before starting
    if not os.path.exists(image_file):
        print(f"Error: Test image file not found at '{image_file}'")
        print("Please ensure the image file exists or set the SIGNATURE_TEST_IMAGE environment variable.")
        # You might want to create a placeholder or download a test image here if needed
        # For now, we exit.
        exit(1) # Exit if file not found

    print(f"Starting Signature endpoint concurrency test for levels {min(concurrency_levels)} to {max(concurrency_levels)}")
    print(f"Target URL: {signature_url}")
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
            futures = [
                executor.submit(send_signature_request, i, image_file, signature_url)
                for i in range(requests_per_level) # Send N requests for this level
            ]

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    # Catch exceptions raised *during* the future's execution if not caught inside send_signature_request
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
    print("\n--- Overall Concurrency Test Summary (Signature Endpoint) ---")
    header_str = f"{'Concurrency':<12} | {'Total Time':<12} | {'Successful':<10} | {'Failed':<8} | {'Avg Time':<12} | {'Min Time':<12} | {'Max Time':<12}"
    print(header_str)
    print("-" * len(header_str)) # Adjust separator length dynamically

    for summary in summary_results:
        print(f"{summary['concurrency']:<12} | {summary['total_time']:<12.4f} | {summary['successful']:<10} | {summary['failed']:<8} | {summary['avg_duration']:<12.4f} | {summary['min_duration']:<12.4f} | {summary['max_duration']:<12.4f}")

    # --- Save Summary to CSV ---
    if summary_results: # Only write if there are results
        # Define fieldnames based on the keys of the first summary dictionary
        fieldnames = list(summary_results[0].keys())
        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write header row
                writer.writeheader()

                # Write data rows
                writer.writerows(summary_results)

            print(f"\nSummary results successfully saved to {csv_filename}")
        except IOError as e:
            print(f"\nError: Could not write summary results to CSV file '{csv_filename}'. Reason: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred while writing to CSV: {e}")
    else:
        print("\nNo summary results generated to save to CSV.")
