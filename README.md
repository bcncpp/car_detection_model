# Car Detection API

A lightweight FastAPI-based server that loads a PyTorch CNN model for car detection.

## Features

- Custom CNN for car detection
- REST API for prediction
- Accepts NumPy array input as nested lists
- Returns model inference results

---

## Requirements

```bash
$pip install fastapi uvicorn torch numpy
```
``
.
├── main.py               # FastAPI app
├── models.py             # CNN model definition
├── cardetection_model.pth  # Trained PyTorch model weights
├── README.md
```
