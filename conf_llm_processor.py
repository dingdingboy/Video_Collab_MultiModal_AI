import numpy as np
from openvino import Core
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json
import torch
from transformers import AutoTokenizer
import os
from modelscope import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

class QwenTextProcessor:
    def __init__(self, model_path: str, device: str = 'NPU'):
        """Initialize Qwen2.5 text processor."""
        self.device = device
        self.model_path = Path(model_path)
        
        # Load OpenVINO model
        self.core = Core()
        model_file = str(self.model_path / "openvino_model.xml")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        self.model = self.core.compile_model(model_file, device)
        self.output_layer = self.model.output(0)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        self.max_length = 2048
        
    def _prepare_input(self, text: str, format_prompt: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """Prepare input for the model."""
        if format_prompt:
            prompt = f"{format_prompt}\n\nText: {text}\n\nSummary:"
        else:
            prompt = f"Please summarize the following text:\n\nText: {text}\n\nSummary:"
            
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        input_ids = inputs["input_ids"].numpy()
        attention_mask = inputs["attention_mask"].numpy()
        
        return input_ids, len(input_ids[0])
    
    def _decode_output(self, output_tensor: np.ndarray, input_length: int) -> str:
        """Decode model output tokens to text."""
        # Convert output logits to token ids
        output_ids = np.argmax(output_tensor, axis=-1)
        
        # Only take generated tokens (skip input tokens)
        generated_ids = output_ids[0][input_length:]
        
        # Decode tokens to text
        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return text.strip()
        
    def summarize(self, text: str, format_prompt: Optional[str] = None) -> Dict[str, str]:
        """Summarize input text according to format prompt."""
        try:
            # Prepare input
            input_ids, input_length = self._prepare_input(text, format_prompt)
            
            # Run inference
            outputs = self.model([input_ids])[self.output_layer]
            
            # Decode output
            summary = self._decode_output(outputs, input_length)
            
            return {
                "original_text": text,
                "summary": summary,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "original_text": text,
                "summary": "",
                "status": "error",
                "error": str(e)
            }

def get_available_devices() -> List[str]:
    """Get list of available OpenVINO devices."""
    core = Core()
    return core.available_devices

def create_processor(model_path: str, device: str = 'NPU') -> QwenTextProcessor:
    """Factory function to create text processor."""
    available_devices = get_available_devices()
    if device not in available_devices:
        raise ValueError(f"Device {device} not available. Available devices: {available_devices}")
        
    return QwenTextProcessor(model_path, device)

# Example usage
if __name__ == "__main__":
    # Test the processor
    model_path = r"C:\Users\test\source\models\Qwen3-8B-nf4-ov"
    text = """
    OpenVINO is an open-source toolkit for optimizing and deploying AI models.
    It supports many frameworks and can run on various hardware platforms including
    CPUs, GPUs, and neural accelerators.
    """
    
    try:
        processor = create_processor(model_path, device="NPU")
        
        # Test basic summarization
        result = processor.summarize(text)
        print("\nBasic Summarization:")
        print(result["summary"])
        
        # Test formatted summarization
        format_prompt = """
        Summarize the text in the following format:
        - Main topic
        - Key points (bullet points)
        - Conclusion
        """
        result = processor.summarize(text, format_prompt=format_prompt)
        print("\nFormatted Summarization:")
        print(result["summary"])
        
    except Exception as e:
        print(f"Error: {str(e)}")