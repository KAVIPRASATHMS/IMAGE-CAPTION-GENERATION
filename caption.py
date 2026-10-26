import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_NAME = "Salesforce/blip-image-captioning-base"

device = "cpu"

print("Loading image captioning model...")

processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=5
        )

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    image_path = input("Enter image path: ").strip()
    caption = generate_caption(image_path)
    print("\nGenerated Caption:", caption)
