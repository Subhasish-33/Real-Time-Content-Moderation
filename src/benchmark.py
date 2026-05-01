import time
import os
import torch
import numpy as np
import evaluate
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
from datasets import load_dataset

def get_model_size(path, is_onnx=False):
    """Calculates model size in MB."""
    size = 0
    if not os.path.exists(path):
        return 0
    if is_onnx:
        for file in os.listdir(path):
            if file.endswith('.onnx'):
                size += os.path.getsize(os.path.join(path, file))
    else:
        # Check for standard PyTorch weights
        for file in os.listdir(path):
            if file.endswith(('.bin', '.safetensors', '.pt')):
                size += os.path.getsize(os.path.join(path, file))
    return size / (1024 * 1024)

def benchmark_model(model, tokenizer, texts, num_passes=1000):
    """Measures average and P95 latency."""
    latencies = []
    
    # Warmup (Important for JIT/ONNX initialization)
    for _ in range(10):
        inputs = tokenizer(texts[0], return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        if isinstance(model, torch.nn.Module):
            with torch.no_grad():
                model(**inputs)
        else:
            model(**inputs)
            
    # Benchmark Loop
    for i in range(num_passes):
        text = texts[i % len(texts)]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        
        start = time.perf_counter()
        if isinstance(model, torch.nn.Module):
            with torch.no_grad():
                model(**inputs)
        else:
            model(**inputs)
        end = time.perf_counter()
        latencies.append((end - start) * 1000) # ms
        
    return {
        "avg_latency": np.mean(latencies),
        "p95_latency": np.percentile(latencies, 95)
    }

def evaluate_precision(model, tokenizer, dataset):
    """Evaluates Precision on a subset of the dataset."""
    metric = evaluate.load("precision")
    all_preds = []
    all_labels = []
    
    for item in dataset:
        # Robust column detection for Jigsaw dataset
        text_column = "comment_text" if "comment_text" in item else "text"
        label_col = "toxic" if "toxic" in item else "label"
            
        inputs = tokenizer(item[text_column], return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        
        if isinstance(model, torch.nn.Module):
            with torch.no_grad():
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
            
        logits = outputs.logits
        if isinstance(logits, torch.Tensor):
            pred = torch.argmax(logits, dim=-1).item()
        else: # ONNX returns numpy-like arrays
            pred = np.argmax(logits, axis=-1)[0]
            
        all_preds.append(pred)
        all_labels.append(item[label_col])
        
    return metric.compute(predictions=all_preds, references=all_labels)["precision"]

def main():
    # --- CONFIGURATION ---
    # Path to the PyTorch model folder
    pytorch_path = "./models/pytorch_distilbert"
    # Path to the ONNX folder
    onnx_quantized_path = "./models/onnx_quantized"
    # The specific filename Antigravity used for the INT8 model
    onnx_file_name = "model_quantized.onnx" 
    
    print("Loading datasets for evaluation...")
    try:
        # Primary choice: Jigsaw
        dataset = load_dataset("jigsaw_toxicity_pred", split="train", trust_remote_code=True)
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        test_split = dataset["test"]
    except:
        # Fallback choice
        dataset = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="test", trust_remote_code=True)
        test_split = dataset
        
    # Select 200 samples for precision (balanced between latency and accuracy measurement)
    eval_dataset = test_split.shuffle(seed=42).select(range(200))
    # Select 50 texts to rotate through for the 1,000-pass latency benchmark
    sample_texts = [item["comment_text" if "comment_text" in item else "text"] for item in eval_dataset.select(range(50))]
    
    # --- PYTORCH BENCHMARK (FP32 Baseline) ---
    print("\n[1/2] Benchmarking PyTorch (FP32) Model...")
    tokenizer_pt = AutoTokenizer.from_pretrained(pytorch_path)
    model_pt = DistilBertForSequenceClassification.from_pretrained(pytorch_path)
    model_pt.to("cpu") # Benchmarking strictly on CPU as per production requirements
    model_pt.eval()
    
    pt_size = get_model_size(pytorch_path, is_onnx=False)
    pt_latencies = benchmark_model(model_pt, tokenizer_pt, sample_texts, num_passes=1000)
    pt_precision = evaluate_precision(model_pt, tokenizer_pt, eval_dataset)
    
    # --- ONNX BENCHMARK (INT8 Optimized) ---
    print("[2/2] Benchmarking ONNX (INT8) Model...")
    tokenizer_onnx = AutoTokenizer.from_pretrained(onnx_quantized_path)
    
    # Explicitly loading the quantized ONNX graph on CPU
    model_onnx = ORTModelForSequenceClassification.from_pretrained(
        onnx_quantized_path, 
        file_name=onnx_file_name,
        provider="CPUExecutionProvider"
    )
    
    onnx_size = get_model_size(onnx_quantized_path, is_onnx=True)
    onnx_latencies = benchmark_model(model_onnx, tokenizer_onnx, sample_texts, num_passes=1000)
    onnx_precision = evaluate_precision(model_onnx, tokenizer_onnx, eval_dataset)
    
    # --- FINAL REPORTING ---
    print("\n" + "="*30)
    print("FINAL PERFORMANCE REPORT")
    print("="*30)
    print(f"PyTorch (FP32) -> Size: {pt_size:>6.2f} MB | Latency: {pt_latencies['avg_latency']:>5.2f} ms | Precision: {pt_precision:.4f}")
    print(f"ONNX (INT8)    -> Size: {onnx_size:>6.2f} MB | Latency: {onnx_latencies['avg_latency']:>5.2f} ms | Precision: {onnx_precision:.4f}")
    
    speedup = pt_latencies['avg_latency'] / onnx_latencies['avg_latency']
    precision_drop = pt_precision - onnx_precision
    
    print("-" * 30)
    print(f"🚀 Speedup Factor: {speedup:.2f}x")
    print(f"📉 Precision Drop: {precision_drop*100:.2f}%")
    print(f"📦 Size Reduction: {(1 - onnx_size/pt_size)*100:.1f}%")
    
    if onnx_latencies['avg_latency'] <= 50:
        print("✅ Success: Model met the <50ms CPU latency target.")
    else:
        print("❌ Warning: Model failed to hit the <50ms CPU latency target.")

    # --- PLOTTING ---
    labels = ['PyTorch (FP32)', 'ONNX (INT8)']
    latency_vals = [pt_latencies['avg_latency'], onnx_latencies['avg_latency']]
    precision_vals = [pt_precision * 100, onnx_precision * 100]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Latency Bar Chart
    color = 'tab:red'
    ax1.set_xlabel('Model Optimization Version')
    ax1.set_ylabel('Avg Latency (ms)', color=color)
    bars = ax1.bar(labels, latency_vals, color=color, alpha=0.6, width=0.4)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=50, color='r', linestyle='--', label='50ms Production Target')
    ax1.legend(loc='upper left')

    # Precision Line Chart
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Precision (%)', color=color)
    ax2.plot(labels, precision_vals, color=color, marker='o', linewidth=3, markersize=10)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([max(0, min(precision_vals)-5), 100])

    plt.title('Production Engineering: Latency vs. Precision (FP32 vs INT8)')
    fig.tight_layout()
    plt.savefig('latency_vs_precision.png')
    print("\nGenerated 'latency_vs_precision.png' for your GitHub results.")

if __name__ == "__main__":
    main()