# Crowd Density Monitoring System

---

## Overview

This project is a **real-time crowd density monitoring system** designed to process live video streams and dynamically estimate crowd levels using deep learning.

It performs person detection using a YOLO-based model and exposes live analytics through a FastAPI backend with a browser-based dashboard.

Unlike notebook-only AI demos, this project focuses on building a complete inference pipeline that handles:

- Live video streaming  
- RTSP instability  
- Runtime camera switching  
- Continuous analytics updates  

The goal was not just to run a model — but to design and implement a robust, end-to-end applied AI system.

---

## Key Features

- Real-time person detection using a YOLO-based deep learning model  
- Support for multiple input sources:
  - Local webcam  
  - RTSP streams  
  - HTTP MJPEG streams  
- Automatic reconnection logic for unstable streams  
- Live crowd count and density estimation  
- REST API endpoints for analytics  
- Dynamic camera switching without restarting the server  
- Clean engine shutdown and thread handling  
- Web-based monitoring dashboard  

---

## System Architecture

---

### 1. Inference Engine

Responsible for:

- Video frame capture  
- Deep learning model inference  
- Bounding box post-processing  
- Crowd count computation  
- Density estimation  
- Stream failure detection  
- Automatic reconnection handling  

The engine is designed to survive:

- Frame drops  
- Decoder failures  
- RTSP timeouts  
- Network interruptions  

without crashing the server.

---

### 2. Backend API (FastAPI)

Provides:

- `GET /video` — Live MJPEG stream  
- `GET /stats` — Real-time analytics  
- `POST /set-camera` — Runtime source switching  
- Engine lifecycle control  

The backend separates inference logic from API routing for clarity and maintainability.

---

### 3. Frontend Dashboard

The browser-based interface:

- Displays processed live video  
- Shows real-time crowd count  
- Updates density and system status  
- Allows runtime camera switching  
- Polls analytics dynamically  

---

## Technology Stack

- Python  
- FastAPI  
- OpenCV  
- YOLO-based deep learning model  
- Uvicorn (ASGI server)  
- HTML + JavaScript frontend  

---

## Design Considerations

---

### Real-World Stream Handling

In real deployments, video streams are unstable.

This system explicitly handles:

- RTSP timeouts  
- HTTP stream interruptions  
- Frame read failures  
- Decoder errors  
- Network jitter  

The engine automatically attempts reconnection without terminating the application.

---

### Separation of Concerns

- Inference engine handles AI and video processing  
- API layer handles communication  
- Frontend handles visualization  

This modular design improves extensibility and scalability.

---

### Runtime Flexibility

Camera sources can be switched dynamically via API calls without restarting the application.

---

## Running the Project Locally

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <project-directory>
```

2. Install Dependencies
```pip install -r requirements.txt```

4. Start the Server
``` uvicorn app.main:app --reload ```
or
```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
6. Open in Browser
``` http://127.0.0.1:8000 ```

---

## Example Use Cases

- Public event monitoring
- Crowd safety analysis
- Retail occupancy tracking
- Smart surveillance systems
- Real-time AI inference deployment practice

---

## What This Project Demonstrates
### This project demonstrates:
- Applied machine learning engineering
- Real-time inference pipeline design
- Robust stream handling
- API-driven architecture
- End-to-end system integration
- Production-style debugging and stability handling
It goes beyond training a model in a notebook and focuses on building a deployable AI system.

---

## Limitations
- Accuracy depends on pretrained model performance
- No custom dataset fine-tuning included
- CPU-based inference may limit FPS on low-end machines
- Designed primarily for applied AI and system integration learning

---

## Potential Improvements
- Multi-camera parallel processing
- Model optimization (quantization or ONNX export)
- GPU acceleration support
- Formal performance evaluation (precision, recall, mAP reporting)
- Threshold-based alert system
- Historical analytics storage
