import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import sys
import os

def run_inference(image_path, model_name="microsoft/trocr-large-handwritten"):
    print(f"Loading model: {model_name}")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"Prediction: {generated_text}")
    return generated_text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_inference.py <image_path> [model_name]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "microsoft/trocr-large-handwritten"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
        
    run_inference(image_path, model_name)
