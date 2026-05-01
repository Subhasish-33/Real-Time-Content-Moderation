import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

app = FastAPI(title="Toxic Content Detection API")

# Global variables for model and tokenizer
tokenizer = None
model = None

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    is_toxic: bool
    confidence: float
    latency_ms: float

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    model_path = "./models/onnx_quantized"
    print(f"Loading ONNX Quantized model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ORTModelForSequenceClassification.from_pretrained(model_path, file_name="model_quantized.onnx")
    print("Model loaded successfully.")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Toxic Content Detection API",
        "usage": "Send a POST request to /evaluate with a JSON body {'text': 'your text here'}",
        "docs": "/docs"
    }

@app.post("/evaluate", response_model=PredictionOutput)
async def evaluate(input_data: TextInput):
    import time
    start = time.perf_counter()
    
    # Preprocess
    inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, max_length=128)
    
    # Inference
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Postprocess
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    pred_label = np.argmax(logits, axis=-1)[0]
    confidence = float(probs[0][pred_label])
    
    end = time.perf_counter()
    
    # Note: Assuming label 1 is toxic. You might need to adjust based on dataset mapping.
    is_toxic = bool(pred_label == 1)
    
    return PredictionOutput(
        is_toxic=is_toxic,
        confidence=confidence,
        latency_ms=(end - start) * 1000
    )
