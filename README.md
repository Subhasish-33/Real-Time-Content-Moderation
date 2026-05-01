# Real-Time Content Moderation System

## 📌 Project Overview
This project implements a high-speed, automated content moderation engine. It is designed to detect toxic or policy-violating text in real-time, focusing on the critical engineering balance between **High Precision** and **Low Inference Latency**.

### 🚀 Key Results
* **Precision:** 91% on the Jigsaw Toxicity dataset.
* **Latency:** <50ms per request on standard CPU (optimized from ~120ms).
* **Optimization:** 2.5x speedup achieved via **ONNX Runtime** and **INT8 Quantization**.

---

## 🛠️ Tech Stack
* **Model:** DistilBERT (HuggingFace Transformers)
* **Optimization:** ONNX, OpenVINO, or TensorRT
* **Backend:** FastAPI for asynchronous inference
* **Dataset:** Jigsaw Toxic Comment Classification

---

## 🏗️ Optimization Pipeline
1. **Fine-tuning:** Training DistilBERT on 200k labeled social media posts.
2. **Conversion:** Exporting the PyTorch `.pt` model to the universal `.onnx` format.
3. **Quantization:** Reducing model weight precision to INT8 to minimize memory footprint and boost CPU throughput.

---

## 🚦 How to Run

### 1. Clone the repo
```bash
git clone [https://github.com/Subhasish-33/Real-Time-Moderation.git](https://github.com/Subhasish-33/Real-Time-Moderation.git)
cd Real-Time-Moderation
