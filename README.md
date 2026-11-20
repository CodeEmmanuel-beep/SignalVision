<div align="center">

# 🚦 SignalVision

**End-to-End Traffic Light Detection & Real-Time Inference Pipeline**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/Ultralytics-YOLO26s-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://docs.ultralytics.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Inference_API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

*Automated dataset preprocessing, multi-variant YOLO26 training and error analysis, and planned containerized inference for traffic signal detection (`Stop`, `Warning`, `Go`) on the LISA Traffic Light Dataset.*

[Overview](#-overview) •
[Dataset & Preprocessing](#-dataset) •
[Validation Results](#-training-run-validation-results) •
[Architecture & Roadmap](#-backend-integration-and-deployment) •
[Project Status](#-project-status)

</div>

---

## Overview

**SignalVision** is a computer vision system for detecting and classifying traffic lights using the **LISA Traffic Light Dataset** and **Ultralytics YOLO26**. Rather than treating model training in isolation, the project covers the full computer vision development lifecycle—from dataset exploration and annotation preprocessing to model evaluation, error investigation, and preparation for backend inference serving.

Three YOLO26 variants (**YOLO26n**, **YOLO26s**, and **YOLO26m**) were trained and investigated. During the **YOLO26s** training run, the model achieved the following validation metrics:

* **Precision:** `94.6%`
* **Recall:** `94.0%`
* **mAP50:** `96.5%`
* **mAP50-95:** `69.1%`

The next phase of work focuses on integrating the selected **YOLO26s** checkpoint into a **FastAPI** service with request validation and rate limiting, containerizing the application with **Docker**, and building out automated testing and deployment.

### Engineering Pipeline
1. Explore and analyze the LISA Traffic Light Dataset.
2. Investigate annotation structure, class distribution, and dataset organization.
3. Build an automated preprocessing pipeline to convert LISA annotations into normalized YOLO format.
4. Train and compare YOLO26 object detection variants (`YOLO26n`, `YOLO26s`, `YOLO26m`).
5. Audit detection errors, false positives, false negatives, confidence thresholds, and NMS IoU behavior.
6. Integrate the selected model into a FastAPI inference service.
7. Containerize, test, and deploy the application.

---

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Dataset Exploration and Analysis](#dataset-exploration-and-analysis)
* [Automated Dataset Preprocessing](#automated-dataset-preprocessing)
* [Model Development](#model-development)
* [Training-Run Validation Results](#training-run-validation-results)
* [Model Evaluation and Investigation](#model-evaluation-and-investigation)
* [Backend Integration and Deployment](#backend-integration-and-deployment)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Project Status](#project-status)
* [Conclusion](#conclusion)

---

## Dataset

SignalVision uses the **LISA Traffic Light Dataset**, which contains driving-scene frames extracted from video sequences along with annotations identifying traffic-light locations and signal states.

### Corrected Dataset Inventory
Following dataset exploration and a preprocessing audit that corrected earlier omissions, the inventory established:

* **~44,075** images across daytime and nighttime driving sequences.
* **111,111+** annotation records, including frames containing multiple traffic lights.
* **7** original annotation categories consolidated into **3** target classes.

> **Preprocessing Audit Note:** During preprocessing, two annotation directories—`sample-nightClip1` and `sample-dayClip6`—were found to have been previously omitted due to a preprocessing issue. The annotation inventory was updated to include them.

### Original Classes to Target Class Mapping
The seven original LISA categories are mapped into three state-based target classes:

| Target ID | Target Class | Original LISA Categories |
| :---: | :--- | :--- |
| `0` | **Stop** | `stop` (Red traffic light), `stopLeft` (Red left-turn arrow) |
| `1` | **Warning** | `warning` (Yellow traffic light), `warningLeft` (Yellow left-turn arrow) |
| `2` | **Go** | `go` (Green traffic light), `goLeft` (Green left-turn arrow), `goForward` (Green forward arrow) |

---

## Dataset Exploration and Analysis

Before training models, exploratory checks were performed to understand data organization and catch quality issues early:

* Inspecting annotation files alongside corresponding image sequences.
* Analyzing distributions across both original and consolidated classes.
* Checking frames with multiple traffic-light annotations per image.
* Reviewing bounding-box coordinates and annotation consistency.
* Examining temporal structure across video-derived sequences.
* Cross-checking annotation records against available image files and visually inspecting annotated and unannotated frames.

### Key Observations
* **Class Imbalance:** `Warning` is substantially underrepresented compared to `Stop` and `Go`—both overall and in the validation split—making per-class metrics essential when evaluating performance.
* **Annotation Coverage:** Manual inspection of a sample of unlabelled images during false-positive analysis revealed visible traffic lights with missing ground-truth labels. Unannotated objects in evaluation data can cause valid detections to be counted as false positives, though this does not mean every background prediction is a true signal.

---

## Automated Dataset Preprocessing

A class-based Python preprocessing pipeline converts raw LISA annotations into the layout and format expected by Ultralytics YOLO:

* Discovers annotation files across dataset directories, accounting for layout differences including `sample-nightClip1` and `sample-dayClip6`.
* Parses annotations, extracts bounding boxes, and maps the 7 source categories into the 3 target classes (`0`, `1`, `2`).
* Converts bounding boxes to normalized YOLO format (`class_id x_center y_center width height`) relative to image width and height.
* Consolidates multiple objects in the same frame into a single label file per image.
* Pairs images with corresponding label files, organizes `train` and `val` splits, and prepares the Ultralytics dataset configuration.

---

## Model Development

Using **Ultralytics YOLO26**, three model sizes were trained and evaluated via automated Python scripts:

* **YOLO26n**
* **YOLO26s** *(selected for the upcoming backend integration phase)*
* **YOLO26m**

The training and experimentation workflow covers dataset preparation, multi-variant training, batch/single-image inference, prediction visualization, confusion matrix generation, confidence and Non-Maximum Suppression (NMS) IoU threshold experiments, and cross-variant comparison.

---

## Training-Run Validation Results

The metrics below were reported on the validation data during the **YOLO26s** training run (not on the training split, a separate standalone validation run, an independent test set, or production traffic).

### Overall Validation Metrics
| Metric | Value |
| :--- | :---: |
| **Validation Images** | `8,815` |
| **Annotated Instances** | `22,615` |
| **Precision** | `94.6%` |
| **Recall** | `94.0%` |
| **mAP50** | `96.5%` |
| **mAP50-95** | `69.1%` |

### Per-Class Validation Metrics
| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stop** | 4,582 | 11,676 | 95.8% | 96.8% | 97.9% | 79.3% |
| **Warning** | 258 | 645 | 94.5% | 91.5% | 95.4% | 57.6% |
| **Go** | 4,076 | 10,294 | 93.5% | 93.7% | 96.1% | 70.6% |
| **All** | **8,815** | **22,615** | **94.6%** | **94.0%** | **96.5%** | **69.1%** |

### Takeaways from Validation Results
* **Stop** leads overall with `96.8%` recall and `79.3%` mAP50-95.
* **Go** achieves strong balance with `93.7%` recall and `70.6%` mAP50-95.
* **Warning** reaches `91.5%` recall and `95.4%` mAP50, but its `57.6%` mAP50-95 shows tighter localization is harder under stricter IoU thresholds—especially given its smaller sample size.
* The gap between **mAP50** and **mAP50-95** reflects the stricter bounding-box overlap requirements across IoU thresholds.

---

## Model Evaluation and Investigation

Beyond headline training-run metrics, evaluation focused on diagnosing error modes:

* Inspecting confusion matrices and class-level errors.
* Analyzing background false positives and missed detections.
* Sweeping confidence thresholds and testing NMS IoU thresholds.
* Comparing behavior across `YOLO26n`, `YOLO26s`, and `YOLO26m`.
* Evaluating how incomplete ground-truth annotations affect error interpretation.

---

## Backend Integration and Deployment

The next milestone exposes **YOLO26s** through a **FastAPI** inference service. These items represent planned work rather than completed features:

1. **FastAPI Model Serving**
   * Load the trained `YOLO26s` weights on application startup.
   * Provide image-upload detection endpoints with format validation and error handling for corrupt or unsupported files.
   * Return structured predictions (class label, confidence score, and bounding-box coordinates) while decoupling inference logic from HTTP route handlers.
2. **API Reliability & Rate Limiting**
   * Enforce request rate limiting and maximum upload size limits.
   * Validate inputs, manage concurrent inference requests safely, and standardize error responses.
3. **Docker Containerization**
   * Package the service and dependencies in Docker, externalize configuration, manage model weights cleanly, and verify inference inside the container.
4. **Testing**
   * Write `pytest` unit tests for preprocessing/inference and integration tests covering valid/invalid uploads, rate limits, concurrency, and containerized runs.
5. **Deployment**
   * Deploy the container with environment configuration, persistent model storage, health checks, and structured logging, then verify live API responses.

---

## Technology Stack

| Area | Tools / Technologies |
| :--- | :--- |
| **Language** | Python |
| **Computer Vision & Deep Learning** | OpenCV, PyTorch, Ultralytics YOLO26 |
| **Data & Preprocessing** | Pandas, NumPy, `pathlib` |
| **Evaluation & Visualization** | Ultralytics validation, Matplotlib |
| **Backend & Serving (Planned)** | FastAPI |
| **Containerization & Testing (Planned)** | Docker, `pytest` |

---

## Project Structure

Conceptual layout combining current training/evaluation artifacts with the planned backend module:

```text
SignalVision
├── LICENSE
├── README.md
├── api
│   └── v1
│       ├── routes
│       └── schemas.py
├── notebooks
│   ├── exploration.ipynb
│   ├── yolo26_s_evaluation.ipynb
│   └── yolo26_s_finetuning.ipynb
├── requirements.txt
├── services
│   ├── evaluation.py
│   ├── image_processing.py
│   └── training.py
└── test_folder
```
