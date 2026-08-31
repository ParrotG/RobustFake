"""Classify the images in ``dataset/detection`` as AI-generated or real.

Run from the project directory with:
    python test.py
"""

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from transformers import CLIPImageProcessor, CLIPVisionModel


PROJECT_DIR = Path(__file__).resolve().parent
CLIP_DIR = PROJECT_DIR / "clip-vit-base-patch32"
MODEL_PATH = PROJECT_DIR / "model" / "clip_binary_classifier_best.pt"
DETECTION_DIR = PROJECT_DIR / "dataset" / "detection"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class FrozenClipBinaryClassifier(nn.Module):
    """The same frozen CLIP vision encoder and classifier head used in training."""

    def __init__(self, vision_encoder: CLIPVisionModel) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        hidden_size = vision_encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_encoder(pixel_values=pixel_values)
        return self.classifier(outputs.pooler_output)


def image_sort_key(path: Path) -> tuple[int, int | str]:
    """Put numeric image names (1.jpg, 2.png, ...) in numeric order."""
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name.lower())


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    if not DETECTION_DIR.is_dir():
        raise FileNotFoundError(f"Detection directory not found: {DETECTION_DIR}")

    image_paths = sorted(
        (path for path in DETECTION_DIR.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=image_sort_key,
    )
    if not image_paths:
        raise RuntimeError(f"No supported images found in: {DETECTION_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    class_to_index = checkpoint["class_to_index"]
    index_to_class = {index: name for name, index in class_to_index.items()}

    image_processor = CLIPImageProcessor.from_pretrained(CLIP_DIR, local_files_only=True)
    vision_encoder = CLIPVisionModel.from_pretrained(CLIP_DIR, local_files_only=True)
    model = FrozenClipBinaryClassifier(vision_encoder).to(device)
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    model.eval()

    with torch.inference_mode():
        for image_path in image_paths:
            with Image.open(image_path) as image:
                pixel_values = image_processor(
                    images=image.convert("RGB"), return_tensors="pt"
                ).pixel_values.to(device)

            predicted_index = model(pixel_values).argmax(dim=1).item()
            predicted_class = index_to_class[predicted_index]
            result = "AI generated" if predicted_class == "FAKE" else "Real"
            print(f"{image_path.name}: {result}")


if __name__ == "__main__":
    main()
