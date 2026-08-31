"""Classify detection images using the model3 probability ensemble.

model3 stores the learned ensemble weight alpha.  It combines the frozen
model1 global classifier and model2 patch-attention classifier probabilities.
"""

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel


PROJECT_ROOT = Path(r"D:\NTU_project\hackson\DINO-v2")
DINO_MODEL_DIR = PROJECT_ROOT / "DINO_v2_based"
ENSEMBLE_PATH = PROJECT_ROOT / "model" / "model3" / "probability_ensemble_alpha.pt"
DETECTION_DIR = PROJECT_ROOT / "dataset" / "detection"
RESULT_PATH = PROJECT_ROOT / "model" / "model3" / "detection_results.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class LocalPatchClassifier(nn.Module):
    """The model2 nonlinear patch-attention head from train3.0.py."""

    def __init__(self, hidden_size: int = 768, num_classes: int = 2):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(patch_tokens), dim=1)
        return self.classifier(torch.sum(weights * patch_tokens, dim=1))


class DetectionDataset(Dataset):
    def __init__(self, image_paths: list[Path], image_transform: transforms.Compose):
        self.image_paths = image_paths
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.image_paths[index]
        with Image.open(path) as image:
            tensor = self.image_transform(image.convert("RGB"))
        return tensor, str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify detection images with model3 ensemble")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def build_transform() -> transforms.Compose:
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_DIR, local_files_only=True)
    return transforms.Compose(
        [
            transforms.Resize(
                processor.size["shortest_edge"], interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop((processor.crop_size["height"], processor.crop_size["width"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )


def main() -> None:
    args = parse_args()
    if not ENSEMBLE_PATH.is_file():
        raise FileNotFoundError(f"Model3 alpha checkpoint not found: {ENSEMBLE_PATH}. Run train4.0.py first.")
    if not DETECTION_DIR.is_dir():
        raise FileNotFoundError(f"Detection directory not found: {DETECTION_DIR}")

    image_paths = sorted(
        path for path in DETECTION_DIR.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported images found in: {DETECTION_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ensemble_checkpoint = torch.load(ENSEMBLE_PATH, map_location=device, weights_only=False)
    alpha = float(ensemble_checkpoint["alpha"])
    class_to_idx: dict[str, int] = ensemble_checkpoint["class_to_idx"]
    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}
    model1_path = Path(ensemble_checkpoint["model1_checkpoint"])
    model2_path = Path(ensemble_checkpoint["model2_checkpoint"])
    if not model1_path.is_file() or not model2_path.is_file():
        raise FileNotFoundError("The model1 or model2 checkpoint referenced by model3 is missing.")

    model1_checkpoint = torch.load(model1_path, map_location=device, weights_only=False)
    model2_checkpoint = torch.load(model2_path, map_location=device, weights_only=False)
    if model1_checkpoint["class_to_idx"] != class_to_idx or model2_checkpoint["class_to_idx"] != class_to_idx:
        raise ValueError("Class label mappings in model1, model2, and model3 do not match.")

    dino = AutoModel.from_pretrained(DINO_MODEL_DIR, local_files_only=True).to(device)
    global_classifier = nn.Linear(model1_checkpoint["feature_dim"], len(class_to_idx)).to(device)
    global_classifier.load_state_dict(model1_checkpoint["classifier_state_dict"])
    patch_classifier = LocalPatchClassifier(
        hidden_size=model2_checkpoint["hidden_size"], num_classes=len(class_to_idx)
    ).to(device)
    patch_classifier.load_state_dict(model2_checkpoint["classifier_state_dict"])
    dino.eval()
    global_classifier.eval()
    patch_classifier.eval()

    data_loader = DataLoader(
        DetectionDataset(image_paths, build_transform()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Using learned alpha = {alpha:.6f}; Pf = alpha * Pg + (1 - alpha) * Pp")

    results: list[dict[str, str | float]] = []
    with torch.inference_mode():
        for images, paths in data_loader:
            images = images.to(device, non_blocking=True)
            hidden_states = dino(pixel_values=images).last_hidden_state
            pg = torch.softmax(global_classifier(hidden_states[:, 0, :]), dim=1)
            pp = torch.softmax(patch_classifier(hidden_states[:, 1:, :]), dim=1)
            pf = alpha * pg + (1.0 - alpha) * pp
            confidences, predictions = pf.max(dim=1)

            for path, prediction, confidence, global_probability, patch_probability, final_probability in zip(
                paths, predictions.tolist(), confidences.tolist(), pg.cpu().tolist(), pp.cpu().tolist(), pf.cpu().tolist()
            ):
                result = {
                    "image": str(Path(path).relative_to(DETECTION_DIR)),
                    "prediction": idx_to_class[prediction],
                    "confidence": round(confidence, 6),
                    "fake_probability": round(final_probability[class_to_idx["FAKE"]], 6),
                    "real_probability": round(final_probability[class_to_idx["REAL"]], 6),
                    "model1_global_probability": round(global_probability[prediction], 6),
                    "model2_patch_probability": round(patch_probability[prediction], 6),
                }
                results.append(result)
                print(f"{result['image']}: {result['prediction']} (confidence: {result['confidence']:.2%})")

    with RESULT_PATH.open("w", newline="", encoding="utf-8-sig") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nClassified {len(results)} image(s). CSV result: {RESULT_PATH}")


if __name__ == "__main__":
    main()
