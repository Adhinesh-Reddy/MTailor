import requests
import base64
import argparse
import time
from pathlib import Path
import os

# ================== CONFIG ==================
CEREBRIUM_URL = "https://api.cortex.cerebrium.ai/v4/p-2ac12dba/mtailor-deployment1/run"
CEREBRIUM_API_KEY = "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTJhYzEyZGJhIiwiaWF0IjoxNzQ4NDYyODg3LCJleHAiOjIwNjQwMzg4ODd9.SuzoCL1C6faEuYQUPCjp9SNoI_4aMF7D6ff4DR5IiqzuqGLlKpSq0bJlIYTYCJy-7pio2TWHVA5zzClgFu7pZpKVLBc9dAWJOTDX3Um5UeFCxrAs9jMXLFQ5l4MktsFa-mLXYm_NkxD-2ALb--0IWhFpcHiQwjhW2I5BHrX0RUTfBbX5HM0eG1Qy7uVQCxLyltpX2bJ0cvIuZJ9dmn3sG_7OHZmAQspqQbcJ_WFMwDtfQqILLF_Kl5jZAu1ala5hNJkaDSS_qEwHz53I_ZtEt3ej_YN6ZQYJjqjC9GfOOdszohvsFuRMEN5TxkpJ26xFRO_ROwGmbTlK4mWY4YDWjA"
 
# ============================================

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def call_cerebrium(image_path: str):
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return

    image_b64 = encode_image_base64(image_path)
    payload = {
        "request": {
            "image_base64": image_b64
        }
    }

    headers = {
    "Authorization": CEREBRIUM_API_KEY,
    "Content-Type": "application/json"
    }

    print("Sending request to Cerebrium...")
    start = time.time()
    response = requests.post(CEREBRIUM_URL, json=payload, headers=headers)
    end = time.time()

    if response.status_code == 200:
        result = response.json()
        class_id = result.get("result", {}).get("class_index")

        print(f"Class ID: {class_id}")
        print(f"Inference Time: {end - start:.3f} seconds")
    else:
        print(f"Request failed: {response.status_code}")
        print(response.text)

def run_custom_tests():
    print("\n Running Custom Platform Tests...\n")

    headers = {
        "Authorization": f"Bearer {CEREBRIUM_API_KEY}",
        "Content-Type": "application/json"
    }

    dummy_img = base64.b64encode(b"invalid_data").decode("utf-8")
    payload = {
        "request": {
            "image_base64": dummy_img
        }
    }

    # Test 1: Error handling
    try:
        print("Test 1: Invalid image input")
        res = requests.post(CEREBRIUM_URL, json=payload, headers=headers)
        assert res.status_code in [400, 422]
        print("Proper error response for invalid input\n")
    except Exception as e:
        print(f"Error handling test failed: {e}")

    # Test 2: Latency benchmark
    try:
        print("Test 2: Average latency")
        times = []
        for _ in range(3):
            start = time.time()
            _ = requests.post(CEREBRIUM_URL, json=payload, headers=headers)
            end = time.time()
            times.append(end - start)
        avg = sum(times) / len(times)
        print(f"Average latency over 3 requests: {avg:.3f} seconds\n")
    except Exception as e:
        print(f"Latency test failed: {e}")

    # Test 3: Unauthorized access
    try:
        print("Test 3: Missing token")
        bad_headers = {"Content-Type": "application/json"}
        res = requests.post(CEREBRIUM_URL, json=payload, headers=bad_headers)
        assert res.status_code == 401
        print("Unauthorized access correctly rejected\n")
    except Exception as e:
        print(f"Auth test failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test Cerebrium ML Deployment")
    parser.add_argument("image", nargs="?", help="Path to image for classification")
    parser.add_argument("--custom-tests", action="store_true", help="Run platform-level tests")

    args = parser.parse_args()

    if not CEREBRIUM_API_KEY:
        print("Missing CEREBRIUM_API_KEY. Set it using:\nexport CEREBRIUM_API_KEY=<your_token>")
        return

    if args.image:
        call_cerebrium(args.image)

    if args.custom_tests:
        run_custom_tests()

if __name__ == "__main__":
    main()
