"""Classify all images in dataset/detection as FAKE or REAL."""

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
CHECKPOINT_PATH = PROJECT_ROOT / "model" / "model1" / "best_classifier.pt"
DETECTION_DIR = PROJECT_ROOT / "dataset" / "detection"
RESULT_PATH = PROJECT_ROOT / "model" / "model1" / "detection_results.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect FAKE/REAL images with the trained DINOv2 classifier")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def build_transform() -> transforms.Compose:
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_DIR, local_files_only=True)
    return transforms.Compose(
        [
            transforms.Resize(
                processor.size["shortest_edge"],
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop((processor.crop_size["height"], processor.crop_size["width"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )


class DetectionDataset(Dataset):
    def __init__(self, image_paths: list[Path], image_transform: transforms.Compose):
        self.image_paths = image_paths
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.image_paths[index]
        with Image.open(path) as image:
            image_tensor = self.image_transform(image.convert("RGB"))
        return image_tensor, str(path)


def main() -> None:
    args = parse_args()
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Trained classifier not found: {CHECKPOINT_PATH}. Run train2.0.py first."
        )
    if not DETECTION_DIR.is_dir():
        raise FileNotFoundError(f"Detection folder not found: {DETECTION_DIR}")

    image_paths = sorted(
        path for path in DETECTION_DIR.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported image files found in: {DETECTION_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    class_to_idx: dict[str, int] = checkpoint["class_to_idx"]
    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}

    dino = AutoModel.from_pretrained(DINO_MODEL_DIR, local_files_only=True).to(device)
    dino.eval()
    classifier = nn.Linear(checkpoint["feature_dim"], len(class_to_idx)).to(device)
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    classifier.eval()

    dataset = DetectionDataset(image_paths, build_transform())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    results: list[dict[str, str | float]] = []
    with torch.inference_mode():
        for images, paths in loader:
            images = images.to(device, non_blocking=True)
            cls_features = dino(pixel_values=images).last_hidden_state[:, 0, :]
            probabilities = torch.softmax(classifier(cls_features), dim=1).cpu()
            confidences, predictions = probabilities.max(dim=1)

            for path, predicted_index, confidence, probability in zip(
                paths, predictions.tolist(), confidences.tolist(), probabilities.tolist()
            ):
                result = {
                    "image": str(Path(path).relative_to(DETECTION_DIR)),
                    "prediction": idx_to_class[predicted_index],
                    "confidence": round(confidence, 6),
                    "fake_probability": round(probability[class_to_idx["FAKE"]], 6),
                    "real_probability": round(probability[class_to_idx["REAL"]], 6),
                }
                results.append(result)
                print(f"{result['image']}: {result['prediction']} (confidence: {result['confidence']:.2%})")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nClassified {len(results)} image(s). Detailed results saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
