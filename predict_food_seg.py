from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torchvision.models.segmentation import deeplabv3_resnet50

SIZE = 256
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = deeplabv3_resnet50(
    weights=None,
    weights_backbone=None,
    num_classes=2
)
model.load_state_dict(torch.load(
    "checkpoints/food_seg_binary.pth",
    map_location=device,
    weights_only=True
))
model.to(device)
model.eval()

image_path = input("Enter food image path: ").strip()
image = Image.open(image_path).convert("RGB")
original_size = image.size

resized = image.resize((SIZE, SIZE))
x = np.asarray(resized, dtype=np.float32) / 255.0
x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x)["out"]
    mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

# Resize the predicted mask to the original image dimensions.
mask_img = Image.fromarray(mask * 255).resize(
    original_size, Image.Resampling.NEAREST
)
mask_img.save("food_mask.png")

# Overlay the predicted food region in red.
base = image.convert("RGBA")
overlay = Image.new("RGBA", original_size, (255, 0, 0, 0))
overlay.putalpha(mask_img.point(lambda p: int(p * 0.45)))
overlay_pixels = overlay.load()
for y in range(original_size[1]):
    for x in range(original_size[0]):
        if mask_img.getpixel((x, y)) > 0:
            overlay_pixels[x, y] = (255, 0, 0, 115)

Image.alpha_composite(base, overlay).convert("RGB").save("food_overlay.jpg")

print("Saved segmentation mask: food_mask.png")
print("Saved overlay: food_overlay.jpg")
