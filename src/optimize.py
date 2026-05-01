import os
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

def optimize_model():
    model_path = "./models/pytorch_distilbert"
    onnx_path = "./models/onnx_base"
    quantized_path = "./models/onnx_quantized"

    print("Loading PyTorch model and exporting to ONNX...")
    # Exporting to ONNX
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ORTModelForSequenceClassification.from_pretrained(model_path, export=True)
    
    # Save the base ONNX model
    tokenizer.save_pretrained(onnx_path)
    model.save_pretrained(onnx_path)
    
    print("Applying Dynamic INT8 Quantization...")
    # Initialize the quantizer
    quantizer = ORTQuantizer.from_pretrained(model)
    
    # Configure dynamic quantization (best for transformer models on CPU)
    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    
    # Apply quantization
    quantizer.quantize(
        save_dir=quantized_path,
        quantization_config=dqconfig,
    )
    # The quantizer saves the model as model_quantized.onnx by default
    
    # Also save the tokenizer to the quantized directory
    tokenizer.save_pretrained(quantized_path)
    print(f"Quantized model saved to {quantized_path}")

if __name__ == "__main__":
    optimize_model()
