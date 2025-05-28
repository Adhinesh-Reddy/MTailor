### `README.md`

```markdown
# MTailor MLOps Assessment – Classification Model Deployment with Cerebrium

This project demonstrates how to build, export, and deploy a classification neural network using ONNX and Cerebrium. The model is containerized using Docker and deployed with GPU support to Cerebrium’s serverless platform.

---

## Deliverables

- PyTorch → ONNX model (with preprocessing embedded)
- Docker-based deployment using `debian:bookworm-slim`
- Cerebrium endpoint for image classification
- API tested using base64 image input
- CI pipeline to verify Docker builds on every push
- CLI test client for real-time API testing

---

## Project Structure

```

.
- Dockerfile               # Custom container for deployment
- main.py                 # Cerebrium entrypoint (run function)
- model.py                # ONNX runtime + preprocessing class
- convert\_to\_onnx.py      # Converts model + preprocessing to ONNX
- classifier.onnx         # Final ONNX model (with preprocessing)
- requirements.txt        # Runtime dependencies
- cerebrium.toml          # Cerebrium deployment config
- test\_server.py          # CLI tool to test Cerebrium endpoint
- .github/workflows/docker-build.yml  # CI pipeline for Docker

````

---

## Setup Instructions

### 1. Create & Activate Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
````

---

### 2. Export Model to ONNX (with preprocessing)

```bash
python convert_to_onnx.py
```

This creates `classifier.onnx` that accepts raw image input.

---

### 3. Deploy to Cerebrium (Docker)

```bash
cerebrium login
cerebrium deploy
```

You’ll receive a live endpoint like:

```
POST https://api.cortex.cerebrium.ai/v4/p-xxxx/mtailor-deployment1/run
```

---

### 4. Set API Token

```bash
export CEREBRIUM_API_KEY=your_token_here
```

---

### 5. Test the API with CLI

```bash
python test_server.py n01667114_mud_turtle.JPEG
```

Output:

```
Class ID: 35
Inference Time: 1.23 seconds
```

Add `--custom-tests` for platform health checks:

```bash
python test_server.py n01667114_mud_turtle.JPEG --custom-tests
```

---

## Optional: Test Docker Image Locally

```bash
docker build -t mtailor-test .
docker run --rm mtailor-test
```

---

## Continuous Integration (CI)

Docker builds are tested automatically via GitHub Actions on every push or PR to `main`.

CI Workflow:

```
.github/workflows/docker-build.yml
```

---

## Model Info

* Format: ONNX
* Input: `uint8` image (224x224)
* Output: class index (int)
* Preprocessing: embedded inside the ONNX model (resize, normalize, NCHW conversion)


## Summary

| Feature                           | Status |
| --------------------------------- | ------ |
| ONNX model with preprocessing     | ✅      |
| Dockerized Cerebrium deployment   | ✅      |
| REST API for image classification | ✅      |
| Real-time testing CLI             | ✅      |
| CI pipeline with Docker build     | ✅      |

---

## Author

Sai Tatireddy
University of Florida — M.S. in Computer Science
Email: [saitatireddy25@gmail.com](mailto:saitatireddy25@gmail.com)
