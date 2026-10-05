import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "nateraw/food"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)
print("Loading food classifier...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

image_path = input("Enter the food image path: ").strip()
image = Image.open(image_path).convert("RGB")

inputs = processor(images=image, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.no_grad():
    logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)[0]

top = torch.topk(probabilities, k=5)

print("\nTop food predictions:")
for score, idx in zip(top.values.tolist(), top.indices.tolist()):
    label = model.config.id2label[idx]
    print(f"{label}: {score:.1%}")
